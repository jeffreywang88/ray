"""AgentX load generation with InferenceX's traffic, sharded over client nodes."""

import glob
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# The dataset and `agentx` preset InferenceX's agentic benchmarks use.
DATASET = "semianalysis_cc_traces_weka_062126"
# Peak memory of one AgentX shard (64 sessions of up to ~1M-token histories plus
# its manager processes), measured at ~3.9 GB.
SHARD_MEMORY_GB = 4
# Per-request record fields kept from profile_export.jsonl.
FIELDS = {
    "isl": "input_sequence_length",
    "osl": "output_sequence_length",
    "ttft_ms": "time_to_first_token",
    "itl_ms": "inter_token_latency",
    "cached": "usage_prompt_cache_read_tokens",
}


@dataclass
class LoadConfig:
    agentx_bin: str
    model: str
    tokenizer: str
    duration_s: float
    seed: int
    # Send X-Session-ID, AgentX's stable per-conversation correlation id (a
    # subagent has its own), so session-aware routers pin a conversation's turns.
    session_ids: bool = True


def agentx_env(cfg: LoadConfig) -> Dict[str, str]:
    return {
        "AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID": str(cfg.session_ids).lower()
    }


def agentx_argv(
    cfg: LoadConfig, urls: List[str], concurrency: int, seed: int, workers: int
) -> List[str]:
    return [
        cfg.agentx_bin,
        "profile",
        "--url",
        *urls,
        "--model",
        cfg.model,
        "--tokenizer",
        cfg.tokenizer,
        "--tokenizer-trust-remote-code",
        "--concurrency",
        str(concurrency),
        "--benchmark-duration",
        str(cfg.duration_s),
        "--random-seed",
        str(seed),
        "--public-dataset",
        DATASET,
        "--workers-max",
        str(workers),
        "--unsafe-override",
        "--ui",
        "none",
    ]


def _metric(record: Dict[str, Any], key: str) -> float:
    value = record.get("metrics", {}).get(key)
    if isinstance(value, dict):
        value = value.get("value")
    return np.nan if value is None else float(value)


def load_records(path: str) -> Dict[str, Any]:
    """Keeps error-free profiling records, as InferenceX does, as numpy columns."""
    columns: Dict[str, List[float]] = {k: [] for k in FIELDS}
    timestamps: Dict[str, List[int]] = {"start_ns": [], "end_ns": []}
    errors = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record.get("metadata", {})
            if metadata.get("benchmark_phase", "profiling") != "profiling":
                continue
            if record.get("error"):
                errors += 1
                continue
            for name in timestamps:
                timestamps[name].append(int(metadata.get(f"request_{name}") or -1))
            for name, key in FIELDS.items():
                columns[name].append(_metric(record, key))
    return {
        "columns": {
            **{k: np.asarray(v, dtype=np.float64) for k, v in columns.items()},
            **{k: np.asarray(v, dtype=np.int64) for k, v in timestamps.items()},
        },
        "errors": errors,
    }


@ray.remote(num_cpus=0)
def run_shard(argv: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    out_dir = tempfile.mkdtemp(prefix="agentx_")
    log_path = os.path.join(out_dir, "agentx.log")
    try:
        with open(log_path, "w") as log:
            rc = subprocess.run(
                [*argv, "--output-artifact-dir", out_dir],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **env},
            ).returncode
        if rc != 0:
            with open(log_path) as log:
                text = log.read()
            # Keep the log on the client node; report the first errors (the
            # cause) and the tail.
            kept = shutil.copy(log_path, f"/tmp/agentx_failed_{os.getpid()}.log")
            first = [line for line in text.splitlines() if " ERROR " in line][:20]
            raise RuntimeError(
                f"agentx exited with {rc} (log kept at {kept}):\n"
                + "\n".join(first)
                + "\n...\n"
                + text[-2000:]
            )
        paths = glob.glob(
            os.path.join(out_dir, "**", "profile_export.jsonl"), recursive=True
        )
        if not paths:
            raise RuntimeError(f"no profile_export.jsonl under {out_dir}")
        return load_records(paths[0])
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def run_load(
    cfg: LoadConfig,
    urls: List[str],
    concurrency: int,
    client_node_ids: List[str],
    client_cpus: float,
    client_memory_gb: float,
) -> Dict[str, Any]:
    """Runs one AgentX shard per engine node and merges their records.

    AgentX gives session i trace i, so one shard per node gives every node the
    same sessions at every cluster size: per-node load doesn't change with N.
    Shards are spread over the client nodes; a client node's shards share its
    seed (and dataset cache), and each run has its own cache-bust prefix, so
    sessions never share prompt prefixes across shards.
    """
    per_url = max(1, round(concurrency / len(urls)))
    per_client = -(-len(urls) // len(client_node_ids))
    if per_client * SHARD_MEMORY_GB > client_memory_gb * 0.8:
        raise ValueError(
            f"{per_client} AgentX shards per client node need ~{per_client * SHARD_MEMORY_GB}"
            f" GB, more than 80% of the client nodes' {client_memory_gb:.0f} GB: add "
            "--client-nodes"
        )
    # AgentX's default is 0.75 x CPUs - 1 workers per process; split that.
    workers = max(1, int(client_cpus * 0.75 - 1) // per_client)
    refs = []
    for i, url in enumerate(urls):
        client = i % len(client_node_ids)
        argv = agentx_argv(cfg, [url], per_url, seed=cfg.seed + client, workers=workers)
        refs.append(
            run_shard.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    client_node_ids[client], soft=False
                )
            ).remote(argv, agentx_env(cfg))
        )
    # A failed shard (AgentX aborts if any warmup request fails) costs only its
    # own records; the phase reports it as an issue.
    shards, shard_errors = [], []
    for i, ref in enumerate(refs):
        try:
            shards.append(ray.get(ref))
        except Exception as e:
            shard_errors.append(f"shard {i} ({urls[i]}): {str(e)[:1000]}")
    if not shards:
        raise RuntimeError("every AgentX shard failed:\n" + "\n".join(shard_errors))
    return {
        "columns": {
            k: np.concatenate([s["columns"][k] for s in shards])
            for k in shards[0]["columns"]
        },
        "errors": sum(s["errors"] for s in shards),
        "shards": len(refs),
        "shard_errors": shard_errors,
        "shard_sizes": [s["columns"]["osl"].size for s in shards],
        # Each shard's own profiling window: shards can start minutes apart.
        "shard_windows_ns": [_shard_window(s["columns"]) for s in shards],
    }


def _shard_window(columns: Dict[str, np.ndarray]) -> Optional[List[int]]:
    """A shard's first request start and last end; None if it has no records."""
    starts, ends = columns["start_ns"], columns["end_ns"]
    starts, ends = starts[starts >= 0], ends[ends >= 0]
    if not starts.size or not ends.size:
        return None
    return [int(starts.min()), int(ends.max())]
