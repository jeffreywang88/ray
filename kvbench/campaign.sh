#!/bin/bash
# Full sweep. For each node count N: scale the cluster to N worker nodes, then for each
# speedup and routers-per-node setting run every variant once per round (interleaved),
# REPS rounds. Runs land in <out>/s<speedup>_r<R>/<variant>_N<N>_rep<k>; finished runs
# are skipped, so rerunning the same command resumes.
#
# usage: campaign.sh <out_dir>
# env:   ENGINE_CPUS (required), NODES="4 16 32 64", SPEEDUPS="1 2 4", ROUTERS_PER_NODE="1 2",
#        REPS=5, DUR=180
set -u
OUT=$1
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
: "${ENGINE_CPUS:?set ENGINE_CPUS to the worker node CPU count}"
NODES=${NODES:-"4 16 32 64"}; SPEEDUPS=${SPEEDUPS:-"1 2 4"}; RPN=${ROUTERS_PER_NODE:-"1 2"}
REPS=${REPS:-5}; DUR=${DUR:-180}
mkdir -p "$OUT"
VARIANTS="nomesh:bench/prom-nomesh mesh_serve:bench/prom-mesh-serve mesh_zmq:bench/prom-mesh-zmq"
for n in $NODES; do
  $KVBENCH_PY "$KVBENCH_DIR/cluster_tools.py" scale "$n" "$ENGINE_CPUS" | tee -a "$OUT/chain.log" || exit 1
  for sp in $SPEEDUPS; do
    for r in $RPN; do
      for rep in $(seq 1 "$REPS"); do
        for v in $VARIANTS; do
          label=${v%%:*}; ref=${v#*:}
          run="$OUT/s${sp}_r${r}/${label}_N${n}_rep${rep}"
          [ -s "$run/prom_summary.json" ] && continue
          rm -rf "$run"
          "$KVBENCH_DIR/point.sh" "$run" "$ref" "$n" "$sp" "$r" "$DUR" \
            || echo "$(date -Is) $run FAILED" >> "$OUT/chain.log"
        done
      done
    done
  done
done
$KVBENCH_PY "$KVBENCH_DIR/cluster_tools.py" release
for d in "$OUT"/s*_r*; do
  $KVBENCH_PY "$KVBENCH_DIR/prom_tables.py" table "$d" > "$d/tables.md"
done
echo "$(date -Is) campaign done" >> "$OUT/chain.log"
