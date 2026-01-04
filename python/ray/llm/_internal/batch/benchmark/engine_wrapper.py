import pandas as pd
import numpy as np
import asyncio
from typing import Any, Dict, Iterator, List

from vllm import LLM
import vllm

from transformers import AutoTokenizer

class Tokenizer:
    """Tokenizer that outputs column-oriented format.
    
    Used by BOTH raw map_batches and processor pipeline for fair comparison.
    The __data wrapping is done separately by DataColumnWrapper (for raw map_batches)
    or by the processor framework (for processor pipeline).
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __call__(self, batch: pd.DataFrame) -> Dict[str, Any]:
        prompts = batch['prompt'].tolist()
        input_ids_list = self.tokenizer(prompts)["input_ids"]
        
        # Return column-oriented format
        return {
            "prompt": prompts,
            "tokenized_prompt": input_ids_list,
        }


class DataColumnWrapper:
    """Converts column-oriented batch to row-oriented format wrapped in __data.
    
    This mimics what the processor framework does internally, allowing raw map_batches
    to use the same input format as vLLMEngineStageSyncUDF.
    """
    
    DATA_COLUMN: str = "__data"
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # Get batch size from first column
        first_key = next(iter(batch.keys()))
        first_val = batch[first_key]
        if hasattr(first_val, "tolist"):
            first_val = first_val.tolist()
        batch_size = len(first_val)
        
        # Convert column-oriented to row-oriented
        rows = []
        for i in range(batch_size):
            row = {}
            for key, val in batch.items():
                if hasattr(val, "__getitem__"):
                    row[key] = val[i]
                else:
                    row[key] = val
            rows.append(row)
        
        return {self.DATA_COLUMN: rows}

class vLLMSyncWrapper:
    """Wrapper that exactly matches StatefulStageSyncUDF's row-oriented format.
    
    Input/Output format: {"__data": [list of row dicts]}
    This matches StatefulStageSyncUDF in base.py and vLLMEngineStageSyncUDF in vllm_engine_stage.py
    """
    
    IDX_IN_BATCH_COLUMN: str = "__idx_in_batch"
    DATA_COLUMN: str = "__data"
    
    def __init__(
        self,
        model_path: str,
        mode: str = "classify",
        output_column: str = "probs",
        max_decode_tokens: int = 100,
        ignore_eos: bool = False,
        std_dev: float = 1.0,
        uniform: bool = False,
        skewed: bool = False,
    ):
        self.model_path = model_path
        self.mode = mode
        self.output_column = output_column
        self.max_decode_tokens = max_decode_tokens
        self.ignore_eos = ignore_eos
        self.std_dev = std_dev
        self.uniform = uniform
        self.skewed = skewed
        
        # Initialize LLM with appropriate task
        self.llm = LLM(
            model=self.model_path,
            enforce_eager=True,
            max_model_len=512,
            runner="pooling",
            convert="classify",
        )

    def _prepare_batch(self, batch: Dict[str, Any]) -> tuple:
        """Prepare batch for processing: extract inputs, assign indices, and separate error rows.

        Args:
            batch: The input batch.

        Returns:
            A tuple of (inputs, normal_rows, error_row_indices).
        """
        # Handle the case where the batch is empty.
        if not batch:
            return [], [], set()

        if self.DATA_COLUMN not in batch:
            raise ValueError(
                f"[Internal] {self.DATA_COLUMN} not found in batch {batch}"
            )

        inputs = batch.pop(self.DATA_COLUMN)
        if hasattr(inputs, "tolist"):
            inputs = inputs.tolist()

        # Assign the index of the row in the batch to the idx_in_batch_column.
        # This is because the UDF output may be out-of-order (if asyncio.as_completed
        # is used internally for example), and we need to carry over unused input
        # columns to the next stage. Thus, we use the row index in batch to match
        # the output of the UDF with the input.
        for idx, row in enumerate(inputs):
            row[self.IDX_IN_BATCH_COLUMN] = idx

        # Separate error rows from normal rows. Error rows (those with
        # __inference_error__ set) bypass the UDF to avoid crashes when
        # expected fields are missing (e.g., generated_tokens for DetokenizeUDF).
        normal_rows = []
        error_row_indices = set()
        for idx, row in enumerate(inputs):
            if row.get("__inference_error__") is not None:
                error_row_indices.add(idx)
            else:
                normal_rows.append(row)

        return inputs, normal_rows, error_row_indices

    def _process_single_output(self, output, inputs, not_outputed_rows):
        """Merge output back into inputs. Matches StatefulStageBaseUDF._process_single_output()."""
        idx_in_batch = output.pop(self.IDX_IN_BATCH_COLUMN)
        not_outputed_rows.remove(idx_in_batch)
        inputs[idx_in_batch].pop(self.IDX_IN_BATCH_COLUMN)
        inputs[idx_in_batch].update(output)

    def udf(self, rows: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Run vLLM on batch. Matches vLLMEngineStageSyncUDF.udf() pattern."""
        if not rows:
            return
        
        # Build prompts - use tokenized_prompt (matching vLLMEngineStageSyncUDF)
        prompts = [
            vllm.inputs.data.TokensPrompt(
                prompt_token_ids=row.get("tokenized_prompt").tolist()
                if hasattr(row.get("tokenized_prompt"), "tolist")
                else row.get("tokenized_prompt"),
            )
            for row in rows
        ]
        
        results = self.llm.encode(
            prompts=prompts,
            pooling_params=vllm.PoolingParams(
                truncate_prompt_tokens=-1,
                task="classify",
            ),
            pooling_task="classify",
        )
        
        # Yield outputs one by one (matching StatefulStageSyncUDF.udf pattern)
        for row, out in zip(rows, results):
            yield {
                self.IDX_IN_BATCH_COLUMN: row[self.IDX_IN_BATCH_COLUMN],
                self.output_column: out.outputs.data.numpy(),
                "prompt": row.get("prompt", ""),
                "prompt_token_ids": row.get("tokenized_prompt"),
                "num_input_tokens": len(row.get("tokenized_prompt")) if row.get("tokenized_prompt") is not None else 0,
            }

    def __call__(self, batch: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Process batch. Exactly matches StatefulStageSyncUDF.__call__().
        
        Input: {"__data": [row1, row2, ...]}
        Output: {"__data": [row1_with_output, row2_with_output, ...]}
        """
        inputs, normal_rows, error_row_indices = self._prepare_batch(batch)
        
        if not inputs and not normal_rows and not error_row_indices:
            # Empty batch case
            yield {}
            return
        
        # Process using udf (generator) like StatefulStageSyncUDF does
        not_outputed_rows = set(range(len(inputs))) - error_row_indices
        if normal_rows:
            for output in self.udf(normal_rows):
                self._process_single_output(output, inputs, not_outputed_rows)
        
        if not_outputed_rows:
            raise ValueError(f"The rows {not_outputed_rows} are not outputed.")
        
        # Clean up idx column from error rows
        for idx in error_row_indices:
            inputs[idx].pop(self.IDX_IN_BATCH_COLUMN, None)
        
        # Return row-oriented format wrapped in __data (matching StatefulStageSyncUDF)
        yield {self.DATA_COLUMN: inputs}


class vLLMSyncWrapperDirect:
    """Direct wrapper without yield pattern overhead - for comparison."""
    
    def __init__(
        self,
        model_path: str,
        mode: str = "classify",
        output_column: str = "probs",
        max_decode_tokens: int = 100,
        ignore_eos: bool = False,
        std_dev: float = 1.0,
        uniform: bool = False,
        skewed: bool = False,
    ):
        self.model_path = model_path
        self.mode = mode
        self.output_column = output_column
        self.max_decode_tokens = max_decode_tokens
        self.ignore_eos = ignore_eos
        self.std_dev = std_dev
        self.uniform = uniform
        self.skewed = skewed
        
        self.llm = LLM(
            model=self.model_path,
            enforce_eager=True,
            max_model_len=512,
            runner="pooling",
            convert="classify",
        )

    def __call__(self, batch: pd.DataFrame):
        input_ids = batch['input_ids'].tolist()
        
        prompts = [
            vllm.inputs.data.TokensPrompt(
                prompt_token_ids=token_id_list.tolist(),
            ) for token_id_list in input_ids
        ]

        if self.mode == "classify":
            result = self.llm.encode(
                prompts=prompts,
                pooling_params=vllm.PoolingParams(
                    truncate_prompt_tokens=-1,
                    task="classify",
                ),
                pooling_task="classify",
            )
            output_data = [out.outputs.data for out in result]
            return {self.output_column: output_data}
        elif self.mode == "generate":
            if self.skewed:
                # Skewed mode: exactly 5 requests get 1000, rest get 10
                num_prompts = len(prompts)
                # Select exactly 5 random indices to get 1000
                indices_1000 = np.random.choice(num_prompts, size=min(5, num_prompts), replace=False)
                sampled_max_tokens = [10] * num_prompts
                for idx in indices_1000:
                    sampled_max_tokens[idx] = 1000
            elif self.uniform:
                # Uniform sampling from (1, 1000)
                sampled_max_tokens = [
                    int(np.random.randint(1, 1001)) for _ in prompts
                ]
            else:
                # Normal distribution sampling
                sampled_max_tokens = [
                    int(np.clip(
                        int(np.random.normal(
                            loc=self.max_decode_tokens,
                            scale=self.std_dev
                        )),
                        1,
                        2000
                    )) for _ in prompts
                ]

            sampling_params_list = [
                vllm.SamplingParams(
                    max_tokens=max_tokens,
                    ignore_eos=self.ignore_eos,
                    temperature=1.0,
                    top_p=1.0,
                ) for max_tokens in sampled_max_tokens
            ]
            
            result = self.llm.generate(
                prompts=prompts,
                sampling_params=sampling_params_list,
            )
            output = {
                self.output_column: [out.outputs[0].text for out in result]
            }
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        
        return output


class vLLMAsyncWrapper:
    def __init__(
        self,
        model_path: str,
        mode: str = "classify",
        output_column: str = "probs",
    ):
        self.model_path = model_path
        self.mode = mode
        self.output_column = output_column
        self.request_id = 0
        
        # Initialize AsyncLLMEngine with appropriate task
        engine_args = vllm.AsyncEngineArgs(
            model=self.model_path,
            enforce_eager=True,
            max_model_len=512,
            runner="pooling",
            convert="classify",
        )
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)

    async def __call__(self, batch: pd.DataFrame) -> dict:
        input_ids = batch['input_ids'].tolist()
        
        # Process all requests concurrently
        tasks = []
        for token_id_list in input_ids:
            prompt = vllm.inputs.data.TokensPrompt(
                prompt_token_ids=token_id_list.tolist(),
            )
            
            pooling_params = vllm.PoolingParams(
                truncate_prompt_tokens=-1,
                task="classify",
            )
            
            request_id = str(self.request_id)
            self.request_id += 1
            
            # Create async task for each request
            task = self._process_single_request(request_id, prompt, pooling_params)
            tasks.append(task)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks)
        
        output = {
            self.output_column: [result for result in results]
        }
        return output
    
    async def _process_single_request(
        self,
        request_id: str,
        prompt: vllm.inputs.data.TokensPrompt,
        pooling_params: vllm.PoolingParams,
    ):
        """Process a single classification request asynchronously."""
        stream = self.engine.encode(
            request_id=request_id,
            prompt=prompt,
            pooling_params=pooling_params,
            truncate_prompt_tokens=pooling_params.truncate_prompt_tokens,
        )
        
        # Consume the stream until the request is finished
        async for request_output in stream:
            if request_output.finished:
                return request_output
        
        raise RuntimeError(
            "[vLLM] The request is not finished. This should not happen."
        )