#!/bin/bash
# One run: install a variant on every node, deploy N engines (one per worker node) with a
# Serve proxy (HAProxy) and R LLMRouter replicas on every node, drive AgentX from the head
# across all engine nodes' proxies for DUR seconds, and collect metrics from every node.
#
# usage: point.sh <run_dir> <variant_branch> <N> <speedup> <routers_per_node> [duration_s]
# env:   ENGINE_CPUS (required: a worker node's CPU count), SPW=28
set -u
RUN=$1; REF=$2; N=$3; SPEEDUP=$4; R=$5; DUR=${6:-180}
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
: "${ENGINE_CPUS:?set ENGINE_CPUS to the worker node CPU count}"
SPW=${SPW:-28}
export RAY_SERVE_INGRESS_ROUTER_REPLICAS_PER_NODE=$R
PY=$KVBENCH_PY; K=$KVBENCH_DIR
mkdir -p "$RUN"; LOG=$RUN/point.log
shutdown() { $PY -c "import ray; from ray import serve; ray.init(address='auto', logging_level='ERROR'); serve.shutdown()" >>"$LOG" 2>&1; }
fail() { echo "FAILED: $*" | tee -a "$LOG"; shutdown; exit 1; }

$PY "$K/install_variant.py" --repo "$KVBENCH_REPO" --ref "$REF" >>"$LOG" 2>&1 || fail "install $REF"
POINT=(--num-gpu-blocks 16384 --max-num-seqs 256 --speedup "$SPEEDUP" --flush-interval-s 0.5
  --block-size 16 --max-num-batched-tokens 8192
  --router-env DYN_RUNTIME_NUM_WORKER_THREADS=8 --router-env BENCH_INSTRUMENT=stats
  --engine-kwargs '{"renderer_num_workers":4}' --replica-env DYN_MOCKER_KV_COALESCE_MS=10)
$PY "$K/app.py" --num-replicas "$N" --engine-cpus "$ENGINE_CPUS" "${POINT[@]}" >"$RUN/deploy.log" 2>&1 || fail "deploy (see deploy.log)"
$PY "$K/cluster_tools.py" placement "$N" "$R" >>"$LOG" 2>&1 || fail "placement"
mapfile -t METRICS < <($PY "$K/cluster_tools.py" metrics-urls)
mapfile -t INGRESS < <($PY "$K/cluster_tools.py" ingress-urls)
[ "${#INGRESS[@]}" = "$N" ] || fail "expected $N ingress URLs, got ${#INGRESS[@]}"
printf '%s\n' "${INGRESS[@]}" > "$RUN/ingress_urls.txt"

$PY - "$RUN/meta.json" <<PYEOF
import json, re, subprocess, sys, datetime, ray
ray.init(address="auto", logging_level="ERROR")
sha = subprocess.check_output(["git", "-C", "$KVBENCH_REPO", "rev-parse", "$REF"]).decode().strip()
routers = int(re.search(r"(\d+) routers", open("$LOG").read().split("placement ok")[-1]).group(1))
json.dump({"variant_ref": "$REF", "ray_commit": sha, "N": $N, "speedup": $SPEEDUP,
  "routers_per_node": $R, "R": routers, "router_cores": 0,
  "sessions_per_worker": $SPW, "conc_sessions": $N * $SPW, "duration_s": $DUR,
  "engine_cpus": "$ENGINE_CPUS", "nodes_alive": sum(n["Alive"] for n in ray.nodes()),
  "point_args": "${POINT[*]}", "start": datetime.datetime.now(datetime.timezone.utc).isoformat()},
  open(sys.argv[1], "w"), indent=1)
PYEOF

URLARGS=(); for u in "${METRICS[@]}"; do URLARGS+=(--url "$u"); done
$PY "$K/promscrape.py" "${URLARGS[@]}" --out "$RUN/prom.jsonl.gz" --interval 5 &
SCRAPE=$!
$PY "$K/monitor.py" --out "$RUN/monitor.jsonl" --interval 10 >"$RUN/monitor.log" 2>&1 &
MON=$!
timeout $((DUR + 3600)) "$KVBENCH_AGENTX" profile \
  --url "${INGRESS[@]}" --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --tokenizer-trust-remote-code --concurrency $((N * SPW)) \
  --input-file "$KVBENCH_DATA" --custom-dataset-type weka_trace \
  --benchmark-duration "$DUR" --trace-idle-gap-cap-seconds 30 \
  --output-artifact-dir "$RUN/agentx" --ui none --unsafe-override \
  --export-level records --workers-max 16 >"$RUN/agentx.log" 2>&1
RC=$?
kill $SCRAPE $MON 2>/dev/null; wait $SCRAPE $MON 2>/dev/null
shutdown
echo "agentx exit $RC" >>"$LOG"
[ -f "$RUN/agentx/profile_export.jsonl" ] && gzip -1 "$RUN/agentx/profile_export.jsonl"
rm -f "$RUN/agentx/profile_export_aiperf_timeslices.json" "$RUN/agentx/aiperf.log"
$PY "$K/prom_tables.py" run "$RUN" >"$RUN/prom_summary.json" 2>>"$LOG" || { rm -f "$RUN/prom_summary.json"; fail "summary"; }
$PY "$K/compact_runs.py" "$RUN" >>"$LOG" 2>&1
echo "$(date -Is) $RUN agentx_rc=$RC" >> "$(dirname "$(dirname "$RUN")")/chain.log"
exit $RC
