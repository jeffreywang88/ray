# AgentX cluster-scale benchmark

Measures how Ray Serve LLM's orchestration (HAProxy, `LLMRouter`, KV-aware routing)
scales. One Dynamo Mocker engine replica is tuned to reproduce an
[InferenceX](https://github.com/SemiAnalysisAI/InferenceX) datapoint's output tok/s
and p90 interactivity. The same replica is then run on every slot of a full
node and of an N-node cluster. With no orchestration overhead, the cluster would
produce `replicas x single-replica tok/s` at the single replica's interactivity.

The load is the same as InferenceX's agentic runs: AgentX's `agentx` preset replaying
`semianalysis_cc_traces_weka_062126`, closed loop, `--concurrency` sessions per replica.
Metrics use InferenceX's definitions:

- **Output tok/s**: output tokens of error-free profiling requests / (last end - first start).
- **p90 interactivity**: 1 / p90 of the per-request inter-token latency (tok/s/user).

## Phases

| Phase     | Nodes | Engines            | Sessions            |
| --------- | ----- | ------------------ | ------------------- |
| `replica` | 1     | 1                  | C                   |
| `node`    | 1     | k = 8 / g          | k x C               |
| `cluster` | N     | N x k              | N x k x C           |

`replica` first calibrates `--speedup` and `--decode-speedup` (skipped when both are
given), and is the reference for the other phases.

## Reproduce an InferenceX datapoint

From the datapoint, take GPUs per replica `g` (TP x DP), concurrency `C`, output
tok/s per GPU `T`, p90 interactivity `I`, the vLLM log's `GPU KV cache size` per DP
rank and the serve flags. Then:

1. One Mocker replica stands for the whole `g`-GPU deployment, so it must produce
   `T x g` output tok/s at p90 inter-token latency `1000 / I` ms.
2. Set `--gpus-per-replica g --concurrency-per-engine C --target-tok-s-per-gpu T
   --target-p90-interactivity I --kv-cache-tokens <KV tokens x DP>
   --max-num-seqs <max-num-seqs x DP> --max-num-batched-tokens <...> --model <model>`.
3. Set `--nodes N`. The cluster's theoretical output is `N x (8 / g) x T x g` tok/s.

The calibration adjusts the decode rate (`speedup x decode-speedup`) until the p90
inter-token latency matches, and `--speedup` (prefill and TTFT) until tok/s matches,
within `--calibration-tolerance`.

**Example: TP2, c1** (86.0 tok/s/GPU, p90 interactivity 436.7, KV 14,164,800 tokens).
One replica must produce 172 tok/s at a 2.29 ms p90 ITL. With `--nodes 128` there are
4 replicas per node, 512 in total, so the theoretical output is 88,064 tok/s:

```bash
python run.py --gpus-per-replica 2 --concurrency-per-engine 1 \
  --target-tok-s-per-gpu 86.0 --target-p90-interactivity 436.7 --target-mean-ttft-ms 212 \
  --kv-cache-tokens 14164800 --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --nodes 128 --client-nodes 1
```

**Default: DEP4, c128** (1,407 tok/s/GPU, p90 interactivity 107.5, KV 41,178,655 tokens
per DP rank). One replica must produce 5,628 tok/s at a 9.30 ms p90 ITL. With 128 nodes
there are 256 replicas, so the theoretical output is about 1.44M tok/s.

## Outputs

`$TEST_OUTPUT_JSON` and `<--output-dir>/result.json` hold:

- `perf_metrics`: per phase, output tok/s (total and per replica), p90 interactivity,
  req/s, mean and p90 TTFT, routing p50 and p99 (HAProxy's time in the ingress router,
  from `ray_serve_haproxy_ingress_router_latency_ms`), and `tok_s_scaling_efficiency`
  (phase tok/s / (engines x replica tok/s)) and `p90_interactivity_ratio` (phase /
  replica).
- `phases`: every metric per phase, including the router count, KV hit ratio and peak
  CPU of the engine and client nodes.
- `speedup`, `decode_speedup` and the `calibration` history.
- `issues`: the run is invalid, and exits 1, if a phase had failed requests or more
  than 0.1 router fallbacks/s, the calibration did not converge, or an engine or client
  node's CPU peaked above `--max-cpu-pct`. A low scaling efficiency is a result, not an issue.

## Running

- Release test: `llm_serve_agentx_cluster_scale` in `release/release_tests.yaml`, with
  `compute_aws.yaml`. The image layer `byod_llm_agentx_cluster_scale.sh` installs AgentX
  in its own venv and the Dynamo wheel with the Python `MockerWorker`.
- Anywhere else: a Ray cluster whose head has no `CPU` resource, engine workers with
  `--node-cpus` CPUs, and optionally `agentx_client` workers. Submit it as a Ray job with
  this directory as the working dir, so replicas on every node can import `mock_engine`:
  `ray job submit --working-dir . -- python run.py ...`
