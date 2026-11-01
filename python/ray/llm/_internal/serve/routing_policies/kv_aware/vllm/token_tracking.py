from typing import Optional, Type

from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from ray.llm._internal.serve.observability.logging import get_logger
from ray.llm._internal.serve.routing_policies.kv_aware.kv_token_tracker import (
    get_llm_router_handle,
)
from ray.llm._internal.serve.routing_policies.kv_aware.lifecycle import (
    LifecycleEventForwarder,
    get_request_lifecycle_owner,
)
from ray.llm._internal.serve.utils.server_utils import get_serve_request_id

logger = get_logger(__name__)


class RequestTokenTracker:
    """Drives the request lifecycle hooks for one ``generate()`` stream."""

    def __init__(
        self,
        forwarder: LifecycleEventForwarder,
        request_id: str,
        owner: str,
        report_decode_progress: bool = False,
    ):
        self._forwarder = forwarder
        self._request_id = request_id
        # Replica id of the LLMRouter that selected this replica.
        self._owner = owner
        self._cumulative = 0
        self._prefill_marked = False
        self._finished = False
        self._report_decode_progress = report_decode_progress

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
            self._forwarder.report_to(
                self._owner, "on_prefill_complete", self._request_id
            )
        if self._report_decode_progress:
            self._forwarder.report_to(
                self._owner,
                "on_decode_progress",
                self._request_id,
                self._cumulative,
            )

    def finish(self) -> None:
        """Report completion exactly once."""
        if not self._finished:
            self._finished = True
            self._forwarder.report_to(
                self._owner, "on_request_completed", self._request_id
            )


def enable_token_tracking(
    engine_cls: Type[AsyncLLM], report_decode_progress: bool = False
) -> Type[AsyncLLM]:
    """Decorator adding KV-router request lifecycle tracking."""

    class TokenTrackingEngine(engine_cls):
        _lifecycle_forwarder: Optional[LifecycleEventForwarder] = None
        _resolve_warned: bool = False
        _report_decode_progress: bool = report_decode_progress

        def _resolve_lifecycle_forwarder(self) -> Optional[LifecycleEventForwarder]:
            if self._lifecycle_forwarder is None:
                try:
                    # The KVTokenTracker lives inside the LLMRouter, so forward
                    # lifecycle events to its deployment handle method.
                    self._lifecycle_forwarder = LifecycleEventForwarder(
                        get_llm_router_handle()
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
            # Only the router that selected this replica booked load for the
            # request, and only its route names an owner.
            owner = get_request_lifecycle_owner()
            # CUMULATIVE repeats output-so-far per chunk; our accounting sums
            # deltas, so skip it rather than over-count. vLLM's OpenAI layer only
            # uses DELTA/FINAL_ONLY (*Request.to_sampling_params):
            # https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai
            forwarder = (
                self._resolve_lifecycle_forwarder() if owner is not None else None
            )
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
                owner,
                report_decode_progress=self._report_decode_progress,
            )
            try:
                async for output in stream:
                    tracker.on_output(output)
                    yield output
            finally:
                tracker.finish()

    return TokenTrackingEngine
