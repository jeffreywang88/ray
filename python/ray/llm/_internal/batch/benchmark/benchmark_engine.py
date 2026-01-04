# import subprocess
# import sys

# subprocess.check_call([
#     sys.executable,
#     "-m",
#     "pip",
#     "install",
#     "datasets",
# ])

import argparse
import random
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import ray
from ray.llm._internal.batch.processor.vllm_engine_proc import vLLMEngineProcessorConfig
from ray.llm._internal.batch.processor.base import ProcessorBuilder
from ray.llm._internal.batch.stages.configs import (
    ChatTemplateStageConfig,
    DetokenizeStageConfig,
    TokenizerStageConfig,
)
from dataset import ShareGPTDataset, OpenThoughtsDataset

from ray.runtime_env import RuntimeEnv
from engine_wrapper import Tokenizer, DataColumnWrapper, vLLMSyncWrapper

@dataclass(slots=True)
class BenchmarkResult:
    mode: str
    batch_size: int
    samples: int
    elapsed_s: float

    @property
    def throughput(self) -> float:
        return self.samples / self.elapsed_s if self.elapsed_s else 0.0

    def show(self) -> None:
        print("\n" + "=" * 60)
        print(f"BENCHMARK - {self.mode}")
        print("=" * 60)
        print(f"Samples     : {self.samples}")
        print(f"Batch size  : {self.batch_size}")
        print(f"Time (s)    : {self.elapsed_s:.2f}")
        print(f"Throughput  : {self.throughput:.2f} req/s")
        print("=" * 60)


def build_processor(config, preprocess=None, postprocess=None, **kwargs):
    """Build a processor using the given config."""
    return ProcessorBuilder.build(
        config,
        preprocess=preprocess,
        postprocess=postprocess,
        **kwargs,
    )


def main(args):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Add Hugging Face token to the runtime environment
    ray.init(runtime_env=RuntimeEnv(env_vars={"HF_TOKEN": ""}))
    model_name = args.model_name
    dataset_size = args.dataset_size

    if args.dataset == "openthoughts":
        dataset = OpenThoughtsDataset(
            dataset_path="/tmp/data/OpenThoughts-114k-math",
            hf_dataset_id="open-r1/OpenThoughts-114k-math",
            hf_split="train",
            seed=0,
            truncate_prompt=args.truncate_prompt,
        )
    else:
        dataset = ShareGPTDataset(
            dataset_path="/tmp/data/Code-feedback-sharegpt-renamed",
            hf_dataset_id="Crystalcareai/Code-feedback-sharegpt-renamed",
            hf_split="train",
            seed=0,
            truncate_prompt=args.truncate_prompt,
        )
    prompts = dataset.sample(dataset_size)

    ds = ray.data.from_items(prompts)

    if args.raw_map_batches:
        # Stage 1: Tokenizer
        ds = ds.map_batches(
            Tokenizer,
            batch_size=args.batch_size,
            zero_copy_batch=True,
            num_cpus=1,
            compute=ray.data.ActorPoolStrategy(size=40),
            batch_format="pandas",
            fn_constructor_kwargs={
                "model_path": model_name,
            },
        )
        # Stage 2: DataColumnWrapper - converts column-oriented to {"__data": [rows]}
        # This mimics what the processor framework does internally
        ds = ds.map_batches(
            DataColumnWrapper,
            batch_size=args.batch_size,
            zero_copy_batch=True,
        )
        # Stage 3: vLLM engine
        ds = ds.map_batches(
            vLLMSyncWrapper,
            batch_size=args.batch_size,
            # Exactly same map_batches kwargs as the processor framework
            zero_copy_batch=True,
            num_gpus=1,
            compute=ray.data.ActorPoolStrategy(
                size=1,
                max_tasks_in_flight_per_actor=4,
            ),
            fn_constructor_kwargs={
                "model_path": model_name,
                "mode": args.mode,
                "output_column": "probs" if args.mode == "classify" else "generated_text",
                "max_decode_tokens": args.max_decode_tokens,
                "ignore_eos": args.ignore_eos,
                "std_dev": args.std_dev,
            },
        )
    else:
        processor_config_kwargs = dict(
            model_source=model_name,
            engine_kwargs=dict(
                distributed_executor_backend="uni",
                tensor_parallel_size=1,
                # gpu_memory_utilization=0.85,
            ),
            task_type=args.mode,
            batch_size=args.batch_size,
            concurrency=4,
            chat_template_stage=ChatTemplateStageConfig(enabled=False),
            tokenize_stage=TokenizerStageConfig(enabled=True),
            detokenize_stage=DetokenizeStageConfig(enabled=False),
        )
        if args.sync_engine:
            processor_config_kwargs["synchronous_engine"] = True

        processor_config = vLLMEngineProcessorConfig(**processor_config_kwargs)

        if args.mode == "classify":
            processor = build_processor(
                processor_config,
                postprocess=lambda row: {
                    "probs": float(row["embeddings"][0])
                    if row.get("embeddings") is not None and len(row["embeddings"]) > 0
                    else None,
                },
            )
        else:
            if args.dataset == "openthoughts":
                # Reasoning-friendly params: let the model think freely.
                # Don't ignore EOS so natural decode-length variance is
                # preserved — short problems finish fast, hard ones produce
                # long chains.  This is what makes async >> sync.
                def _reasoning_preprocess(row):
                    return dict(
                        prompt=row["prompt"],
                        sampling_params={
                            "max_tokens": args.max_decode_tokens,
                            "ignore_eos": args.ignore_eos,
                            "temperature": 0.6,
                            "top_p": 0.95,
                        },
                    )
                preprocess_fn = _reasoning_preprocess
            else:
                def _default_preprocess(row):
                    return dict(
                        prompt=row["prompt"],
                        sampling_params={
                            "max_tokens": int(np.clip(
                                np.random.normal(
                                    loc=args.max_decode_tokens,
                                    scale=args.std_dev
                                ),
                                1,
                                10000
                            )),
                            "ignore_eos": True,
                            "temperature": 1.0,
                            "top_p": 1.0,
                        },
                    )
                preprocess_fn = _default_preprocess

            processor = build_processor(
                processor_config,
                preprocess=preprocess_fn,
                postprocess=lambda row: {
                    "generated_text": row.get("generated_text", ""),
                },
            )
        ds = processor(ds)

    start = perf_counter()
    ds = ds.materialize()
    elapsed = perf_counter() - start
    
    result = BenchmarkResult(
        mode=args.mode,
        batch_size=args.batch_size,
        samples=dataset_size,
        elapsed_s=elapsed,
    )
    result.show()
    
    print(ds.take(1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark async vLLM engine")
    parser.add_argument(
        "--raw-map-batches",
        action="store_true",
        default=False,
        help="Use raw map batches",
    )
    parser.add_argument(
        "--sync-engine",
        action="store_true",
        default=False,
        help="Use synchronous vLLM engine",
    )
    # Dataset-specific arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "openthoughts"],
        help="Dataset to use. 'openthoughts' uses open-r1/OpenThoughts-114k-math "
             "for reasoning workloads with high decode-length variance.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=2560,
        help="Dataset size",
    )
    parser.add_argument(
        "--truncate-prompt",
        type=int,
        default=512,
        help="Maximum prompt length in characters",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512, # Need this to achieve max throughput for generation
        # default=65536, # Need this to achieve max throughput for classification
        help="Batch size for processing",
    )
    # Model-specific arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="classify",
        choices=["classify", "generate"],
        help="Mode to run the benchmark in",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="HuggingFaceTB/fineweb-edu-classifier",
        help="Model name or path",
    )
    # Generation-specific arguments
    parser.add_argument(
        "--max-decode-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (only for generate mode). "
             "For reasoning workloads (openthoughts), use 4096+ to allow "
             "full chain-of-thought reasoning.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        default=False,
        help="Ignore EOS token during generation (only for generate mode)",
    )
    parser.add_argument(
        "--std-dev",
        type=float,
        default=0.0,
        help="Standard deviation for the normal distribution of the decode length for generation",
    )
    args = parser.parse_args()
    main(args)
