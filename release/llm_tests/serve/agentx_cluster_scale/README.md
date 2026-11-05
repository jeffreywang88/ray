# AgentX cluster-scale benchmark

Measures how Ray Serve LLM's orchestration (HAProxy, `LLMRouter`, KV-aware routing)
scales. One Dynamo Mocker engine replica is tuned to reproduce an
[InferenceX](https://github.com/SemiAnalysisAI/InferenceX) datapoint's output tok/s
and p90 interactivity. The same replica is then run on every slot of a full
node and of N-node clusters, for each N in `--nodes`. With no orchestration
overhead, each cluster would produce `replicas x single-replica tok/s` at the
single replica's interactivity and TTFT.

The load is the same as InferenceX's agentic runs: AgentX's `agentx` preset replaying
`semianalysis_cc_traces_weka_062126`, closed loop, `--concurrency` sessions per replica.
Metrics use InferenceX's definitions:

- **Output tok/s**: output tokens of error-free profiling requests / the window's length.
- **p90 interactivity**: 1 / p90 of the per-request inter-token latency (tok/s/user).

**Metrics cover the window when every shard is profiling.** A shard starts profiling
when its own warmup ends, and AgentX's warmup waits for every primed session to return.
Under load that tail varies by minutes between shards: 920 s apart for 128 nodes. So
the headline metrics use only requests that start between the last shard's first
request and the first shard's last request (`shard_overlap_s`), when the whole cluster
carries profiling traffic. With one shard that is its whole profile. `--duration-s`
must exceed the shards' start skew (`shard_start_skew_s`) by the window you want; the
`all_` metrics use every profiling request, with tok/s summed over each shard's own
window.

**Every node gets the same traffic at every N.** AgentX gives session `i` trace `i`, so
the harness runs one AgentX shard per engine node, each with that node's sessions.
Node 1 of a 128-node cluster replays the same traces as the single node; only the
token content (each client node's seed) and the cache-bust prefix differ, so no two
sessions share a prompt prefix. Without this, larger clusters would replay more
distinct traces per node, and per-node tok/s would change with the trace mix, not with
the orchestration. Shards are spread over `--client-nodes`, whose CPUs split AgentX's
worker processes.

**Each conversation stays on one engine.** AgentX sends `X-Session-ID` with its stable
per-conversation id (`AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID`; a subagent has its
own). HAProxy forwards the header to the `LLMRouter`, and `KVAwareRouter` pins the
session's turns to the engine it first picked. The pin is soft: it moves when Serve
excludes that engine, and expires after 30 idle minutes. `--no-session-ids` turns it
off. Before the load, each phase probes the pinning end to end: 4 sessions per engine
(at most 64), 4 turns each with unrelated prompts, first without and then with a
session id, while every engine names itself in an `x-kvbench-engine` response header.
Every probe session with an id must keep all its turns on one engine. Under load the
engines count requests by whether they served the session before:
`session_new_engine_share` is then the share of requests that start a session, which
doesn't change with N, plus any moved sessions.

**Load starts once Serve has settled.** Each replica that becomes ready reloads its
node's HAProxy, and a reloaded-away HAProxy process force-closes its connections after
Serve's `hard-stop-after` (120 s). AgentX's warmup primers (a session's whole history,
up to ~1M tokens) can run longer than that. So each phase waits for every engine and
router replica to be RUNNING, plus 30 s, before starting AgentX. Between phases it
waits for Serve's actors to be gone, so a timed-out `serve.shutdown()` can't kill
the next phase's controller.

## What the engines stand in for

`mock_engine.MockerServer` replaces `LLMServer` + vLLM. KV-aware routing sees the same
thing from it as from a vLLM replica:

- **KV events:** the Mocker publishes vLLM-format events on ZMQ (content-hashed blocks with
  their token ids), so every router indexes it as it would a vLLM replica.
- **Prompt:** the engine runs the token ids the router hashed. They normally arrive over the
  router's token channel. When they don't, the engine tokenizes the request with the router's
  own vLLM renderer, as vLLM's OpenAI server does. That happens when the router could not
  tokenize (truncated body), when its push failed (its first push to each engine always fails,
  since the ZMQ socket is not connected yet), or when the payload had not arrived. A prompt the
  router never hashed would miss every block the router indexed for the session and add
  blocks no request shares, so the KV hit ratio would fall with the number of
  router × engine pairs.
- **Lifecycle events:** reported with the installed Ray's protocol: to the selecting router
  over ZMQ when it names itself in the `kv-lifecycle-owner` header (replica mesh,
  ray-project/ray#67791), otherwise broadcast to every `LLMRouter` replica.

**Routing up to a 1M-token context.** The router must get, and tokenize, every prompt in
full, as in production. A body HAProxy truncates, or a router call that times out, is routed
without tokens, so without KV overlap. The traces' prompts have a median of 142k tokens and
reach 990k. Two HAProxy limits are raised:

| Flag | Default | Why |
| --- | --- | --- |
| `--router-body-buffer-mib` | 16 | A 1M-token AgentX prompt is ~4.2 MB of JSON: 3.97 bytes per token, measured with the default tokenizer on AgentX's corpus. |
| `--router-timeout-s` | 30 | The router renders and tokenizes a 1M-token prompt in ~3.9 s on an idle core. Serve's default is 5 s. |

Every phase counts truncated bodies, and a run with any is invalid. Serve's router-latency
histogram tops out at 1 s, while routing a long prompt takes seconds (mostly tokenization). So
`route_share_over_top_bucket` reports the share past 1 s, and a routing percentile that falls
there is `None` rather than 1000 ms.

## What the KV hit ratio means

The hit ratio is not a calibrated knob like the speedups. The Mocker runs a block-level
prefix cache on the token ids it is given, and its mechanics match vLLM v1's
`KVCacheManager`:

- Blocks are content-hash chains (16-token blocks).
- The lookup happens at the first schedule. Blocks become visible as their prefill chunk is
  scheduled.
- `cached_tokens` is capped at `(prompt - 1)` rounded down to whole blocks.
- Freed blocks are evicted LRU, a request's tail first, out of `--kv-cache-tokens`.
- Preemption works the same way.

The hit then sets prefill cost, which the timing model charges only for uncached tokens. So the
ratio measures what KV-aware routing controls: whether each turn lands on the engine that holds
its prefix.

It is not production vLLM's hit ratio for the default model:

- **The Mocker can't cache hybrid models.** It prefix-caches a single full-attention KV cache
  and refuses prefix caching with multiple KV-cache groups. DeepSeek-V4's sliding-window and
  compressed groups can only lower vLLM's hit, so read the Mocker's ratio as an upper bound
  for it.
- **Each engine pools a whole DP group's cache.** One Mocker stands for a DEP replica with
  the summed KV cache. Ray Serve LLM runs each DP rank as its own replica with its own cache,
  so the routers in production index `gpus-per-replica` times more workers.
- **Output blocks never hit.** Their token ids are synthetic. With real engines they wouldn't
  either here, because AgentX replays the trace's assistant text, not the generated output.
- **Long prompts cost too little.** The default timing model prices prefill by uncached
  tokens, with no context-length term. That shifts residency and KV pressure, not the cache
  mechanics.

## Phases

| Phase          | Nodes | Engines            | Sessions            |
| -------------- | ----- | ------------------ | ------------------- |
| `replica`      | 1     | 1                  | C                   |
| `node`         | 1     | k = 8 / g          | k x C               |
| `cluster_n<N>` | N     | N x k              | N x k x C           |

`--phases` picks from `replica`, `node` and `cluster`; `cluster` runs once per size in
`--nodes 16,32,64,128`. `--calibration-phase` (`replica` by default, or `node`) first
calibrates `--speedup` and `--decode-speedup` (skipped when both are given; with only
`--speedup`, just the decode rate). `replica` is the scaling reference when it runs,
else `node`. With the speedups given, `--phases node,cluster` skips calibration and
the single node is the reference.

## Reproduce an InferenceX datapoint

From the datapoint, take GPUs per replica `g` (TP x DP), concurrency `C`, output
tok/s per GPU `T`, p90 interactivity `I`, mean and p90 TTFT, the vLLM log's
`GPU KV cache size` per DP rank and its KV block size (`Setting kv cache block size`),
and the serve flags. Then:

1. One Mocker replica stands for the whole `g`-GPU deployment, so it must produce
   `T x g` output tok/s at p90 inter-token latency `1000 / I` ms.
2. Set `--gpus-per-replica g --concurrency-per-engine C --target-tok-s-per-gpu T
   --target-p90-interactivity I --target-mean-ttft-ms <...> --target-p90-ttft-ms <...>
   --kv-cache-tokens <KV tokens x DP> --block-size <KV block size>
   --max-num-seqs <max-num-seqs x DP> --max-num-batched-tokens <...> --model <model>`.
   The block size sets how many blocks the engines publish and every router indexes,
   so it drives the routers' memory and CPU.
3. Set `--nodes N1,N2,...`. Each cluster's theoretical output is
   `N x (8 / g) x T x g` tok/s.

The calibration adjusts the decode rate (`speedup x decode-speedup`) until the p90
inter-token latency matches, and `--speedup` (prefill) until p90 TTFT matches, within
`--calibration-tolerance`. Output tok/s is a check, not a target: AgentX replays
recorded think times, so a closed loop's request rate barely depends on engine speed,
and a run much shorter than InferenceX's hour holds fewer long outputs. TTFT also
includes the router's prompt rendering and tokenization (seconds for the longest
prompts), which the speedups don't touch. If that alone exceeds the target, either
calibrate the engine's own TTFT (`--calibration-ttft engine`: request arrival at the
engine to its first token) to the target, which leaves out tokenization the real
engine also did, or pass `--speedup` to fix prefill speed; then only the decode rate
is calibrated, on p90 ITL.

**Example: TP2, c1** (86.0 tok/s/GPU, p90 interactivity 436.7, KV 14,164,800 tokens).
One replica must produce 172 tok/s at a 2.29 ms p90 ITL. With `--nodes 128` there are
4 replicas per node, 512 in total, so the theoretical output is 88,064 tok/s:

```bash
python run.py --gpus-per-replica 2 --concurrency-per-engine 1 \
  --target-tok-s-per-gpu 86.0 --target-p90-interactivity 436.7 --target-mean-ttft-ms 212 \
  --kv-cache-tokens 14164800 --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --nodes 128 --client-nodes 1
```

**Default: TP4 with MTP, c32**
([InferenceX run 40269923234](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40269923234),
vLLM v0.32.0 on B200: 370.3 tok/s/GPU, p90 interactivity 234.3, mean / p90 TTFT
166 / 267 ms, KV 41,870,178 tokens in 128-token blocks, `--max-num-seqs 128`,
`--max-num-batched-tokens 8192`, 97.7% prefix-cache hits). One replica must produce
1,481 tok/s at a 4.27 ms p90 ITL; a node runs 2 replicas (2,962 tok/s). With 128 nodes
there are 256 replicas, so the theoretical output is about 379k tok/s.

**DEP4, c128** (same run family: 1,407 tok/s/GPU, p90 interactivity 107.5, KV
41,178,655 tokens per DP rank): `--concurrency-per-engine 128 --target-tok-s-per-gpu
1406.97 --target-p90-interactivity 107.5 --target-mean-ttft-ms 1048
--kv-cache-tokens 164714620 --max-num-seqs 512`.

## Outputs

`$TEST_OUTPUT_JSON` and `<--output-dir>/result.json` hold:

- `perf_metrics`: per phase, output tok/s (total and per replica), p90 interactivity,
  req/s, mean and p90 TTFT, routing p50 and p99 (HAProxy's time in the ingress router,
  from `ray_serve_haproxy_ingress_router_latency_ms`; `None` past its 1 s top bucket) and
  `route_share_over_top_bucket`, the engines' own p90 TTFT (`engine_p90_ttft_ms`, from
  `kvbench_mocker_ttft_ms`), KV hit ratio, `tok_s_scaling_efficiency` (phase tok/s /
  (engines x reference tok/s per replica)), `p90_interactivity_ratio` and
  `p90_ttft_ratio` (phase / reference). The reference is `replica`, else `node`.
- `phases`: every metric per phase, including the router count, KV hit ratio, peak
  CPU of the engine and client nodes, HAProxy's `truncated_bodies`, and where the engines'
  prompts came from (`prompts_from_token_channel`, `prompts_engine_tokenized_no_key`,
  `prompts_engine_tokenized_store_miss`, `engine_tokenized_share`), the session probe
  (`session_probe_*`), and the engines' session counts (`session_requests_new`,
  `session_requests_repeat`, `requests_without_session`, `session_new_engine_share`).
- `speedup`, `decode_speedup`, the `calibration` history (also appended to
  `calibration.jsonl` point by point), and `calibration_vs_target`: the calibration
  phase's relative error on tok/s per replica, p90 interactivity, mean and p90 TTFT,
  and the engines' p90 TTFT.
- `issues`: the run is invalid, and exits 1, if a phase had failed requests or failed
  AgentX shards (AgentX aborts a shard if any warmup request fails; the other shards'
  records still count, and `shard_errors` holds the first errors), more
  than 0.1 router fallbacks/s, any truncated request body, AgentX shards that profiled
  together for less than 120 s (`shard_overlap_s`; raise `--duration-s`), failed probe
  requests, or with session ids a probe session that changed engines or engine requests
  without a session id, the calibration did not converge, or an engine or client node's CPU peaked above
  `--max-cpu-pct`. A low scaling efficiency is a result, not an issue.
- `<phase>_records.npz` in `--output-dir`: every profiling request's lengths, TTFT,
  ITL, cached tokens, start and end, with the shard sizes and windows.

## Running

- Release test: `llm_serve_agentx_cluster_scale` in `release/release_tests.yaml`, with
  `compute_aws.yaml`. The image layer `byod_llm_agentx_cluster_scale.sh` installs AgentX
  in its own venv and the Dynamo wheel with the Python `MockerWorker`, pinned by URL and
  sha256 under `s3://air-example-data/rayllm-ossci/dynamo/295c175/`.
- Anywhere else: a Ray cluster whose head has no `CPU` resource, engine workers with
  `--node-cpus` CPUs, and optionally `agentx_client` workers. Submit it as a Ray job with
  this directory as the working dir, so replicas on every node can import `mock_engine`:
  `ray job submit --working-dir . -- python run.py ...`
  - Without an `agentx_client` node type, `--client-resource CPU` takes
    `--client-nodes` plain engine-type workers as load generators; they never get
    engines, so they run no proxy or router. AgentX builds its dataset (~7 min, 6 GB)
    once per node and seed and caches it; `--client-node-ids` pins the client nodes
    across runs so the cache stays warm. Keep AgentX off a small head: its dataset
    and per-session histories take several GB.
  - `--reserve-nodes` (default: the largest phase) keeps that many engine nodes
    requested for the whole run, so the autoscaler starts them during calibration
    and doesn't remove them between phases.
  - `--calibration-start S,D` starts the search near the answer. The Mocker's decode
    step is `5.74 + 54.01 u - 25.74 u^2` ms at KV utilization `u`, divided by
    `S x D`; prefill is `4.21e-7 T^2 + 0.0152 T + 16.5` ms for `T` uncached tokens,
    divided by `S`.
