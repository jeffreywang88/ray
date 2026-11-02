# KV routing replica-mesh benchmark on a multi-node Ray cluster

You are on the **head node** of an autoscaling Ray cluster. Compare three versions of
Ray Serve LLM's KV-aware router, sweeping:
- **engine nodes:** 4, 16, 32, 64
- **mock-engine speedup:** 1×, 2×, 4×
- **LLMRouter replicas per node:** 1, 2

Run each setting 5 times and report the tables `prom_tables.py` produces. Everything
you need is in `kvbench/` on the bench branches of https://github.com/jeffreywang88/ray.

Do the steps in order. Don't change workload settings. If a check fails, stop and
report it; don't work around it.

## 1. What you're comparing

| Label | Branch | What it is |
|---|---|---|
| `nomesh` | `bench/prom-nomesh` | Ray master: engines send lifecycle events to every router over Ray RPC |
| `mesh_serve` | `bench/prom-mesh-serve` | Routers share state over Dynamo's replica mesh; engines report to the owning router via a pinned Serve handle |
| `mesh_zmq` | `bench/prom-mesh-zmq` | Same mesh; engines push lifecycle events to the owning router over ZMQ |

- All three branch from Ray master `fb7c757706` and change only pure-Python files under
  `python/ray/serve` and `python/ray/llm`.
- Each one also widens the event-loop-lag histogram buckets to `[1..10, 25, 50]` ms.
  That change is for benchmarking only.

**Layout of one run** (`point.sh` handles all of this):

**Every worker node runs** one Dynamo mock engine (`MockerServer`), a Serve proxy
(HAProxy, direct streaming), and R `LLMRouter` replicas (`RAY_SERVE_INGRESS_ROUTER_REPLICAS_PER_NODE`).
- The engine reserves all the node's CPUs, so the autoscaler adds one node per engine.
- Each HAProxy only uses the routers on its own node. If Serve also puts a proxy on the
  head, the head gets R routers too: they join the mesh but get no client traffic.

**The head node runs** AgentX and the metric collectors:
- AgentX replays Claude Code traces at 28 sessions per engine (closed loop, recorded
  think times) for 180 s after warmup.
- It spreads requests round-robin across every engine node's proxy (`--url` once per
  node), the way an external load balancer would.

**Engines:** block 16, 16,384 KV blocks, 256 max sequences, polynomial timing scaled by
the speedup. Not a real model or GPU.

## 2. Cluster requirements (check before anything else)

1. **The same Ray on every node, autoscaled ones included.** That's Ray at master commit
   `fb7c757706aafd22025a2103268586cf138a143d` with the `serve` and `llm` extras on
   Python 3.12. If the image doesn't already have it, it needs this wheel:
   `https://s3-us-west-2.amazonaws.com/ray-wheels/master/fb7c757706aafd22025a2103268586cf138a143d/ray-3.0.0.dev0-cp312-cp312-manylinux2014_x86_64.whl`
   Check it:
   `python -c "import ray; print(ray.__commit__)"`
   If it's wrong, stop. The image or cluster config has to change, and you can't fix that from the head node.
2. **Site-packages must be writable by the Ray user on every node.** `install_variant.py`
   writes the variant's files into the installed `ray` package.
3. **Head node:** 16 or more cores. Its Ray `CPU` resource must be lower than a worker's,
   ideally 0, so no engine lands on the head. The dashboard must be running, because
   that's what serves each node's Prometheus endpoint.
4. **Worker node type:** at least 8 CPUs (each node holds one engine plus up to 2
   routers), max workers ≥ 64. Set `ENGINE_CPUS` to its CPU count (a whole number).
5. **Networking:** the head must reach every node on port 8000 (HAProxy) and on its
   metrics port. Nodes must reach each other on all TCP ports (ZMQ).
6. **Internet access** from the head node to GitHub and Hugging Face.
7. **Build tools on the head node:** `git`, `clang`, `protoc`, `cmake`, `pkg-config`,
   OpenSSL headers. On Ubuntu:
   `sudo apt-get install -y clang protobuf-compiler cmake pkg-config libssl-dev`

## 3. One-time setup (head node)

```bash
git clone https://github.com/jeffreywang88/ray.git ~/ray-kv && cd ~/ray-kv
git fetch origin bench/prom-nomesh bench/prom-mesh-serve bench/prom-mesh-zmq
for b in bench/prom-nomesh bench/prom-mesh-serve bench/prom-mesh-zmq; do git branch -f $b origin/$b; done
git checkout bench/prom-mesh-zmq          # any of the three; kvbench/ is identical on all
export KVBENCH_PY=$(which python)          # the Python that has the cluster's Ray
source ~/ray-kv/kvbench/env.sh
python -m venv ~/agentx-venv && ~/agentx-venv/bin/pip install agentx-harness==1.0.7
~/ray-kv/kvbench/prepare_data.sh           # downloads 393 traces, checks md5, writes ~/kvbench-data/cc-k1
~/ray-kv/kvbench/build_dynamo.sh           # 20-40 min; writes ~/kvbench-wheels/ai_dynamo_runtime-*.whl
```

**Why the dataset is prepared:** `prepare_data.sh` takes AgentX's public dataset
(`semianalysisai/cc-traces-weka-062126`) and shrinks each hash block from 64 tokens to 1;
hash ids, prefix sharing, timing and output lengths stay the same. AgentX can't do this
itself, and unscaled prompts (mean 219k tokens) make routers mostly tokenize, not route.

**About the Dynamo build:** `build_dynamo.sh` builds `jeffreywang88/dynamo` at `dc9d9a0fec`
(branch `kvbench-mocker`, which adds the Python `MockerWorker`), with clang because GCC 15+
builds of its bundled libzmq crash on `bind`. `install_variant.py` installs the wheel on any
node missing it at the start of every run, so autoscaled nodes get it too.

**In every new shell:**
```bash
export KVBENCH_PY=$(which python) ENGINE_CPUS=<worker CPU count>
source ~/ray-kv/kvbench/env.sh
```

## 4. Smoke test (about 20 min). Do it before the campaign.

```bash
NODES=4 SPEEDUPS=4 ROUTERS_PER_NODE="1 2" REPS=1 DUR=120 ~/ray-kv/kvbench/campaign.sh ~/kvbench-results/smoke
```

That's 6 runs (3 variants × 1 and 2 routers per node) at 4× speed and N=4. Pass criteria
per run (`~/kvbench-results/smoke/s4_r<R>/<variant>_N4_rep1/`):
- `point.log` has `installed ... dynamo <version>` and `placement ok: 4 engines on 4 worker
  nodes; <M> routers (<R> on each of <P> proxy nodes ...)`.
- `ingress_urls.txt` lists 4 distinct node IPs.
- `prom_summary.json` has `failed` = 0 and `requests` > 0.
  These fields must be non-null: `route_ms_p50`, `router_cpu_ms_per_req_prom`,
  `loop_lag_ms_p99`, `ttft_p99_ms`, `hit_ratio`.
  `loop_lag_samples` must be > 100.
- `mesh_serve` and `mesh_zmq` end with `tracked_requests_end` = 0.
  `nomesh` may leak a few hundred bookings during warmup. That's a known bug on master,
  so record it but don't treat it as a failure.

If a run fails, read `deploy.log`, `point.log`, and `agentx.log` in its folder.
See section 8.

## 5. Full campaign

```bash
nohup ~/ray-kv/kvbench/campaign.sh ~/kvbench-results/main > ~/kvbench-results/main.log 2>&1 &
```

**What `campaign.sh` does:**
- For each N, it scales the cluster to N worker nodes and waits for them.
- Then for each speedup (1, 2, 4) and routers-per-node setting (1, 2), it runs
  `nomesh`, `mesh_serve`, `mesh_zmq` once per round for 5 rounds. Interleaving keeps
  slow drift out of the comparison.
- Each run reinstalls its variant on every node, restarts Serve, deploys, runs AgentX,
  collects metrics, and shuts Serve down.
- Runs land in `~/kvbench-results/main/s<speedup>_r<R>/<variant>_N<N>_rep<k>`; progress in `chain.log`.
- At the end it writes one `tables.md` per `s<speedup>_r<R>` folder. To rebuild them any time:
  `$KVBENCH_PY ~/ray-kv/kvbench/prom_tables.py table ~/kvbench-results/main/s4_r2`

**Size and time:** 3 variants × 4 node counts × 3 speedups × 2 router settings × 5 reps is
360 runs. At about 6 min each, that's about 36 h.
- To split or reorder, set `NODES`, `SPEEDUPS`, `ROUTERS_PER_NODE`, `REPS` or `DUR`
  (e.g. `SPEEDUPS=4 ROUTERS_PER_NODE=2` first).
- Finished runs (those with a `prom_summary.json`) are skipped, so rerunning the same
  command resumes after a crash or fills in failures.

Never run two runs at once; they share ports and nodes. Make sure every cell ends with 5 runs.

## 6. Metrics (all computed by `prom_tables.py`)

- **Measurement window:** from 60 s after AgentX's profiling phase starts to the end of
  the run. Warmup is excluded.
- **Cells:** median over the 5 runs, with [min–max].
- **Scraping:** `promscrape.py` scrapes every node's metrics endpoint every 5 s (with
  `~/.ray/auth_token` if set) and sums across nodes; a round where any node fails is dropped.

| Table | Source |
|---|---|
| Time spent routing p50/p99 (ms) | `ray_serve_haproxy_ingress_router_latency_ms`: each HAProxy's wall time around its router call, all nodes |
| Router CPU per routed request (ms) | `ray_component_cpu_percentage{Component=~".*LLMRouter"}` summed over all routers ÷ client req/s |
| Router event-loop lag p50/p99 (ms), share > 5 ms / > 10 ms | `ray_serve_event_loop_scheduling_latency_ms{deployment="LLMRouter",loop_type="main"}` |
| Steady-state TTFT p50/p99 (ms) | AgentX per-request records (`agentx/ttft.csv.gz`) |
| Context: req/s, fallbacks/s, failed requests, calls into routers per routed request | AgentX; `ray_serve_haproxy_ingress_router_{requests,fallbacks}_total`; `ray_serve_deployment_request_counter_total` |
| Context: KV cache hit ratio, load-view mismatch (%) | `monitor.py` polling every router's `bench_stats` and every engine's `get_stats` every 10 s |
| Context: total LLMRouter replicas | the placement check |

## 7. Correctness checks to run and report

1. **Right code:** each run's `meta.json` `ray_commit` equals
   `git -C ~/ray-kv rev-parse <branch>` for its label.
2. **Install:** every `point.log` shows the same digest for the same variant across runs.
3. **Complete:** 5 runs per cell, 0 failed requests, `loop_lag_samples` > 100 in every run.
4. **No leaks in the mesh variants:** `tracked_requests_end` = 0 in every `mesh_*` run.
5. **Placement:** `placement ok` in every run. The router total is the same across
   variants for the same N and R.
6. **Load generator:** the head's CPU (`host_cpu_pct` in `monitor.jsonl`) stays below
   about 85%. If it doesn't, AgentX may be the bottleneck. Say so next to those cells.
7. **Fallbacks:** `fallbacks_per_s` above about 0.1 means routing failed and HAProxy
   bypassed the router. Report it.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `installed ray is X, need fb7c...` | Wrong Ray in the image (section 2.1). Stop. |
| `nodes differ: ... dynamo={None}` | No wheel found: check `$KVBENCH_DYNAMO_WHEEL` and `~/kvbench-wheels`. |
| `PermissionError` in install | Site-packages isn't writable by the Ray user (section 2.2). |
| `head node has N CPUs >= engine CPUs` | Set `ENGINE_CPUS` to the worker size, or give the head fewer `CPU` resources. |
| `scale` times out | The autoscaler can't add nodes: check the worker type's max count, quotas, and `ray status`. |
| `placement check failed` | Read the message: an engine on the head, an engine node without a proxy, or the wrong router count per node. |
| Deploy times out, `Address already in use` | A replica from a previous run is still alive. Run `serve shutdown -y`, wait 30 s, rerun. |
| AgentX connection errors | The head can't reach `<node>:8000`: check security groups or firewall. |
| `prom_summary.json` missing, `summary` failed | No complete scrape round: try `curl` on each URL from `cluster_tools.py metrics-urls`, with `-H "Authorization: Bearer $(cat ~/.ray/auth_token)"`. |
| `hit_ratio` is null | `monitor.log` shows the error. Usually the routers didn't load the stats hook: check that `~/kvbench_node` exists on the worker nodes. |

## 9. What to send back

1. Every `~/kvbench-results/main/s*_r*/tables.md`.
2. A tarball of `~/kvbench-results/main` without `prom.jsonl.gz` files (they're large).
3. The cluster's specs: instance types and CPU counts for head and workers, the image,
   and the cluster provider.
4. The results of each section 7 check, with any run that failed and why.
5. The output of `ray status` at N=64.
