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
  from `ray_serve_haproxy_ingress_router_latency_ms`; `None` past its 1 s top bucket) and
  `route_share_over_top_bucket`, KV hit ratio, `tok_s_scaling_efficiency` (phase tok/s /
  (engines x replica tok/s)) and `p90_interactivity_ratio` (phase / replica).
- `phases`: every metric per phase, including the router count, KV hit ratio, peak
  CPU of the engine and client nodes, HAProxy's `truncated_bodies`, and where the engines'
  prompts came from (`prompts_from_token_channel`, `prompts_engine_tokenized_no_key`,
  `prompts_engine_tokenized_store_miss`, `engine_tokenized_share`).
- `speedup`, `decode_speedup` and the `calibration` history.
- `issues`: the run is invalid, and exits 1, if a phase had failed requests, more
  than 0.1 router fallbacks/s or any truncated request body, the calibration did not
  converge, or an engine or client node's CPU peaked above `--max-cpu-pct`. A low scaling
  efficiency is a result, not an issue.

## Running

- Release test: `llm_serve_agentx_cluster_scale` in `release/release_tests.yaml`, with
  `compute_aws.yaml`. The image layer `byod_llm_agentx_cluster_scale.sh` installs AgentX
  in its own venv and the Dynamo wheel with the Python `MockerWorker`, pinned by URL and
  sha256 under `s3://air-example-data/rayllm-ossci/dynamo/295c175/`.
- Anywhere else: a Ray cluster whose head has no `CPU` resource, engine workers with
  `--node-cpus` CPUs, and optionally `agentx_client` workers. Submit it as a Ray job with
  this directory as the working dir, so replicas on every node can import `mock_engine`:
  `ray job submit --working-dir . -- python run.py ...`
