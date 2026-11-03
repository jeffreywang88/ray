"""AgentX load generation with InferenceX's traffic, sharded over client nodes."""

import glob
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# The dataset and `agentx` preset InferenceX's agentic benchmarks use.
DATASET = "semianalysis_cc_traces_weka_062126"
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


def agentx_argv(
    cfg: LoadConfig, urls: List[str], concurrency: int, seed: int
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
def run_shard(argv: List[str]) -> Dict[str, Any]:
    out_dir = tempfile.mkdtemp(prefix="agentx_")
    log_path = os.path.join(out_dir, "agentx.log")
    try:
        with open(log_path, "w") as log:
            rc = subprocess.run(
                [*argv, "--output-artifact-dir", out_dir],
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        if rc != 0:
            with open(log_path) as log:
                tail = log.read()[-4000:]
            raise RuntimeError(f"agentx exited with {rc}:\n{tail}")
        paths = glob.glob(
            os.path.join(out_dir, "**", "profile_export.jsonl"), recursive=True
        )
        if not paths:
            raise RuntimeError(f"no profile_export.jsonl under {out_dir}")
        return load_records(paths[0])
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def run_load(
    cfg: LoadConfig, urls: List[str], concurrency: int, client_node_ids: List[str]
) -> Dict[str, Any]:
    """Runs one AgentX shard per client node and merges their records.

    Each shard drives a slice of the URLs with a matching share of the sessions
    and its own seed; a single shard uses ``cfg.seed`` like InferenceX.
    """
    num_shards = min(len(client_node_ids), len(urls))
    per_url = concurrency / len(urls)
    refs = []
    for i in range(num_shards):
        shard_urls = urls[i::num_shards]
        argv = agentx_argv(
            cfg,
            shard_urls,
            concurrency=max(1, round(per_url * len(shard_urls))),
            seed=cfg.seed + i,
        )
        refs.append(
            run_shard.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    client_node_ids[i], soft=False
                )
            ).remote(argv)
        )
    shards = ray.get(refs)
    return {
        "columns": {
            k: np.concatenate([s["columns"][k] for s in shards])
            for k in shards[0]["columns"]
        },
        "errors": sum(s["errors"] for s in shards),
        "shards": num_shards,
    }
