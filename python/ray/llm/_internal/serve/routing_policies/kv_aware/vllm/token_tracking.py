import asyncio
import math
import os
from typing import Any, List, Optional, Type

from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from ray import serve
from ray.actor import ActorHandle
from ray.exceptions import RayActorError, RayTaskError
from ray.llm._internal.serve.observability.logging import get_logger
from ray.llm._internal.serve.routing_policies.kv_aware.kv_aware_actor import (
    get_worker_id,
)
from ray.llm._internal.serve.utils.server_utils import get_serve_request_id

logger = get_logger(__name__)


def _get_prompt_token_ids(prompt: Any) -> List[int]:
    """The prompt's pre-tokenized token ids."""
    try:
        return list(prompt["prompt_token_ids"])
    except (KeyError, TypeError) as e:
        raise ValueError(
            "KV-aware token tracking requires a pre-tokenized prompt "
            f"(dict with 'prompt_token_ids'); got {type(prompt).__name__}"
        ) from e


class LifecycleEventForwarder:
    """Ordered, non-blocking bridge from the engine to the KV router actor
    which maintains per-replica token load statistics.

    ``report`` only enqueues locally, so generation never blocks on the actor.
    A single delivery task per replica drains the queue to the actor, awaiting
    one ``on_lifecycle_events`` call at a time so events arrive in the order they
    were reported. Events that pile up during a call are sent together in the
    next one.
    """

    def __init__(self, actor: ActorHandle, worker_id: int):
        self.actor = actor
        self.worker_id = worker_id
        self._events: asyncio.Queue = asyncio.Queue()
        self._delivery_task: Optional[asyncio.Task] = None
        # KVAwareRouter8: coalesce for at least this long before each delivery,
        # bounding the handle-RPC rate (default 5 ms -> <=200 RPC/s per engine
        # process) instead of draining per loop wakeup under load.
        self._flush_s = (
            float(os.environ.get("KV_LIFECYCLE_FLUSH_MS", "5")) / 1000.0
        )

    def report(self, method_name: str, *args) -> None:
        if self._delivery_task is None or self._delivery_task.done():
            self._delivery_task = asyncio.get_running_loop().create_task(
                self._deliver()
            )
        self._events.put_nowait((method_name, args))

    async def _deliver(self) -> None:
        while True:
            # Wait for the next event, coalesce for the flush interval, then
            # drain whatever queued up behind it into the same batch.
            batch = [await self._events.get()]
            if self._flush_s > 0:
                await asyncio.sleep(self._flush_s)
            while not self._events.empty():
                batch.append(self._events.get_nowait())
            try:
                await self.actor.on_lifecycle_events.remote(batch)
            except (RayActorError, RayTaskError) as e:
                logger.warning("Dropping KV lifecycle events: %s", e)
            finally:
                for _ in batch:
                    self._events.task_done()

    async def flush(self) -> None:
        """Wait until every reported event has been delivered."""
        await self._events.join()

    def close(self) -> None:
        """Cancel the delivery task on engine shutdown."""
        if self._delivery_task is not None:
            self._delivery_task.cancel()
            self._delivery_task = None


class RequestTokenTracker:
    """Drives the request lifecycle hooks for one ``generate()`` stream."""

    def __init__(
        self,
        forwarder: LifecycleEventForwarder,
        request_id: str,
        prompt_token_ids: List[int],
        expected_output_tokens: Optional[int],
        send_token_ids: bool = False,
        block_size: Optional[int] = None,
    ):
        self._forwarder = forwarder
        self._request_id = request_id
        self._cumulative = 0
        self._prefill_marked = False
        self._finished = False
        # KVAwareRouter8: booking is block-granular on the selector side
        # (on_decode_progress books one add_output_block per crossed
        # ceil((prompt+cum)/block_size) boundary), so only chunks that CROSS a
        # boundary need to be reported — filtering here produces a booking
        # sequence identical to per-chunk reporting while cutting the event
        # volume ~block_size x. None = report every chunk (kv5 behavior).
        self._block_size = block_size
        self._prompt_tokens = len(prompt_token_ids)
        self._last_blocks = (
            math.ceil(self._prompt_tokens / block_size) if block_size else 0
        )
        # kv4 (single ingress replica): only the prompt token *count* crosses the
        # wire — on_request_added replays the cached selection by id.
        # kv5 (N ingress replicas, send_token_ids=True): the event may land on a
        # replica that did not do the select, so also forward the prompt token ids
        # to book the reservation self-contained by worker_id.
        added_args = [
            request_id,
            forwarder.worker_id,
            len(prompt_token_ids),
            expected_output_tokens,
        ]
        if send_token_ids:
            added_args.append(list(prompt_token_ids))
        forwarder.report("on_request_added", *added_args)

    def on_output(self, output: RequestOutput) -> None:
        """Observe one engine ``RequestOutput`` (forwarded to the caller as-is).

        vLLM streams either DELTA chunks or a single FINAL_ONLY chunk; both
        carry only new tokens, so output progress simply accumulates.
        """
        step_tokens = sum(len(o.token_ids or []) for o in output.outputs)
        if step_tokens == 0:
            # No new tokens this step (e.g. a finish-only chunk).
            return
        self._cumulative += step_tokens
        if not self._prefill_marked:
            # The first output token signals prefill completion.
            self._prefill_marked = True
            self._forwarder.report("on_prefill_complete", self._request_id)
        if self._block_size:
            # KVAwareRouter8: report only chunks that cross a KV-block boundary
            # (the selector books per crossed block; sub-block progress is a
            # no-op there). The cumulative value at a crossing chunk is the
            # same one per-chunk reporting would deliver, so bookings match.
            new_blocks = math.ceil(
                (self._prompt_tokens + self._cumulative) / self._block_size
            )
            if new_blocks <= self._last_blocks:
                return
            self._last_blocks = new_blocks
        self._forwarder.report("on_decode_progress", self._request_id, self._cumulative)

    def finish(self) -> None:
        """Report completion exactly once."""
        if not self._finished:
            self._finished = True
            self._forwarder.report("on_request_completed", self._request_id)


def enable_token_tracking(
    engine_cls: Type[AsyncLLM], send_token_ids: bool = False
) -> Type[AsyncLLM]:
    """Decorator adding KV-router request lifecycle tracking.

    ``send_token_ids`` (kv5, >1 ingress replica): forward the prompt token ids
    with on_request_added so the reservation can be booked self-contained on any
    ingress replica's selector (its pending selection is local, does not
    propagate). Off for kv4 (single replica), which replays the cached selection.
    """

    class TokenTrackingEngine(engine_cls):
        _lifecycle_forwarder: Optional[LifecycleEventForwarder] = None
        _resolve_warned: bool = False
        _send_token_ids: bool = send_token_ids

        def _resolve_lifecycle_forwarder(self) -> Optional[LifecycleEventForwarder]:
            if self._lifecycle_forwarder is None:
                try:
                    # KVAwareRouter4/5: the KVRouterActor lives in-process in the
                    # LLMRouter, so forward lifecycle events to its deployment
                    # handle method rather than a named KVRouterActor. A
                    # DeploymentHandle supports handle.on_lifecycle_events.remote().
                    from ray.llm._internal.serve.routing_policies.kv_aware.inprocess_actor import (  # noqa: E501
                        get_llm_router_handle,
                    )

                    actor = get_llm_router_handle()
                    worker_id = get_worker_id(
                        serve.get_replica_context().replica_id.unique_id
                    )
                    self._lifecycle_forwarder = LifecycleEventForwarder(
                        actor, worker_id
                    )
                except Exception as e:
                    # Warn once: resolution is retried per request until it succeeds.
                    if not self._resolve_warned:
                        self._resolve_warned = True
                        logger.warning("KV token tracking disabled: %s", e)
            return self._lifecycle_forwarder

        def shutdown(self, *args, **kwargs):
            if self._lifecycle_forwarder is not None:
                self._lifecycle_forwarder.close()
            return super().shutdown(*args, **kwargs)

        async def generate(self, prompt, sampling_params, request_id, *args, **kwargs):
            stream = super().generate(
                prompt, sampling_params, request_id, *args, **kwargs
            )
            forwarder = self._resolve_lifecycle_forwarder()
            # CUMULATIVE repeats output-so-far per chunk; our accounting sums
            # deltas, so skip it rather than over-count. vLLM's OpenAI layer only
            # uses DELTA/FINAL_ONLY (*Request.to_sampling_params):
            # https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai
            if forwarder is None or (
                sampling_params.output_kind == RequestOutputKind.CUMULATIVE
            ):
                async for output in stream:
                    yield output
                return

            lifecycle_request_id = get_serve_request_id() or request_id
            tracker = RequestTokenTracker(
                forwarder,
                lifecycle_request_id,
                _get_prompt_token_ids(prompt),
                # The request's own output cap is its expected length; weights
                # the selection service's decode-block decay.
                # TODO(jeffreywang): Use an agent-provided expected-OSL hint for
                # more accurate decode-load estimation.
                sampling_params.max_tokens,
                send_token_ids=self._send_token_ids,
                # KVAwareRouter8: the engine knows its own KV block size; the
                # tracker uses it to report only block-boundary crossings.
                block_size=getattr(
                    getattr(self.vllm_config, "cache_config", None),
                    "block_size",
                    None,
                ),
            )
            try:
                async for output in stream:
                    tracker.on_output(output)
                    yield output
            finally:
                tracker.finish()

    return TokenTrackingEngine
