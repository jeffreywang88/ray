import argparse

import numpy as np
import ray
from ray.data.llm import build_processor, vLLMEngineProcessorConfig
from ray.llm._internal.batch.stages.configs import (
    ChatTemplateStageConfig,
    DetokenizeStageConfig,
    TokenizerStageConfig,
)
from dataset import ShareGPTDataset
from sync_engine_wrapper import Tokenizer, vLLMSyncWrapper, vLLMAsyncWrapper

from ray.runtime_env import RuntimeEnv


def main(args):
    # Add Hugging Face token to the runtime environment
    ray.init(runtime_env=RuntimeEnv(env_vars={}))
    model_name = args.model_name
    dataset_size = args.dataset_size

    dataset = ShareGPTDataset(
        dataset_path="/tmp/data/Code-feedback-sharegpt-renamed",
        seed=0,
        hf_dataset_id="Crystalcareai/Code-feedback-sharegpt-renamed",
        hf_split="train",
        truncate_prompt=512,
    )
    prompts = dataset.sample(dataset_size)

    ds = ray.data.from_items(prompts)
    # ds = ds.repartition(8)

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

    if args.sync_engine:
        ds = ds.map_batches(
            vLLMSyncWrapper,
            batch_size=args.batch_size,
            batch_format="pandas",
            zero_copy_batch=True,
            num_gpus=1,
            compute=ray.data.ActorPoolStrategy(size=1),
            fn_constructor_kwargs={
                "model_path": model_name,
                "mode": args.mode,
                "output_column": "probs" if args.mode == "classify" else "generated_text",
                "max_decode_tokens": args.max_decode_tokens,
                "ignore_eos": args.ignore_eos,
                "std_dev": args.std_dev,
                "uniform": args.uniform,
                "skewed": args.skewed,
            },
        )
    elif args.async_engine:
        ds = ds.map_batches(
            vLLMAsyncWrapper,
            batch_size=args.batch_size,
            batch_format="pandas",
            zero_copy_batch=True,
            num_gpus=1,
            compute=ray.data.ActorPoolStrategy(size=1),
            fn_constructor_kwargs={
                "model_path": model_name,
                "mode": args.mode,
            },
        )
    else:
        processor_config = vLLMEngineProcessorConfig(
            model_source=model_name,
            engine_kwargs=dict(
                enforce_eager=True,
                max_model_len=512,
            ),
            task_type=args.mode,  # "classify" or "generate"
            batch_size=args.batch_size,
            concurrency=1,
            chat_template_stage=ChatTemplateStageConfig(enabled=False),
            tokenize_stage=TokenizerStageConfig(enabled=False),
            detokenize_stage=DetokenizeStageConfig(enabled=False),
        )

        if args.mode == "classify":
            processor = build_processor(
                processor_config,
                preprocess=lambda row: dict(
                    prompt=row['prompt'],
                    tokenized_prompt=row['input_ids'],
                    pooling_params={
                        "truncate_prompt_tokens": -1,
                    }
                ),
                postprocess=lambda row: {
                    "probs": float(row['embeddings'][0])
                    if row.get('embeddings') is not None and len(row['embeddings']) > 0
                    else None,
                },
            )
        else:  # generate mode
            # Sample decode length per row
            def preprocess_with_sampled_decode_length(row):
                if args.skewed:
                    # Skewed mode: probabilistic approach to get ~5 per batch with 1000
                    # Using probability calibrated for typical batch sizes
                    # For batch_size=1000, prob=5/1000=0.005; for batch_size=100, prob=5/100=0.05
                    # Using a fixed probability that works reasonably across batch sizes
                    prob_1000 = 2.0 / args.batch_size if args.batch_size > 2 else 1.0
                    sampled_max_tokens = 1000 if np.random.random() < prob_1000 else 10
                elif args.uniform:
                    # Uniform sampling from (1, 1000)
                    sampled_max_tokens = int(np.random.randint(1, 1001))
                else:
                    # Normal distribution sampling
                    sampled_max_tokens = int(np.clip(
                        int(np.random.normal(
                            loc=args.max_decode_tokens,
                            scale=args.std_dev
                        )),
                        1,
                        2000
                    ))
                return dict(
                    prompt=row['prompt'],
                    tokenized_prompt=row['input_ids'],
                    sampling_params={
                        "max_tokens": sampled_max_tokens,
                        "ignore_eos": args.ignore_eos,
                        "temperature": 1.0,
                        "top_p": 1.0,
                    }
                )

            processor = build_processor(
                processor_config,
                preprocess=preprocess_with_sampled_decode_length,
                postprocess=lambda row: {
                    "generated_text": row.get('generated_text', ''),
                },
            )
        ds = processor(ds)

    ds = ds.materialize()
    print(ds.take(1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark async vLLM engine")
    parser.add_argument(
        "--sync-engine",
        action="store_true",
        default=False,
        help="Use synchronous vLLM engine instead of async processor",
    )
    parser.add_argument(
        "--async-engine",
        action="store_true",
        default=False,
        help="Use asynchronous vLLM engine instead of synchronous processor",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="HuggingFaceTB/fineweb-edu-classifier",
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=52680800,
        help="Dataset size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        # default=512, # Need this to achieve max throughput for generation
        default=526808, # Need this to achieve max throughput for classification
        # default=2048, 
        # default=526808,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="classify",
        choices=["classify", "generate"],
        help="Mode to run the benchmark in. classify: classify the prompt, generate: generate the response",
    )
    parser.add_argument(
        "--max-decode-tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to generate (only for generate mode)",
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
    parser.add_argument(
        "--uniform",
        action="store_true",
        default=False,
        help="Sample decode length uniformly from (1, 1000) instead of normal distribution",
    )
    parser.add_argument(
        "--skewed",
        action="store_true",
        default=False,
        help="Skewed mode: most requests have 10 decode length, exactly 5 per batch have 1000",
    )
    args = parser.parse_args()
    main(args)
