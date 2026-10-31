import asyncio
import copy
from typing import Any, Dict, List, Optional, Union, get_args

import jinja2
from pydantic import ValidationError
from vllm import envs
from vllm.entrypoints.anthropic.protocol import (
    AnthropicEffort,
    AnthropicMessagesRequest,
)
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.launchers.cli_args import FrontendArgs
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.exceptions import VLLMClientError, VLLMValidationError
from vllm.renderers import renderer_from_config
from vllm.renderers.chat_utils import load_chat_template
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.renderers.online_renderer import OnlineRenderer

from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from ray.llm._internal.serve.engines.vllm.vllm_engine import (
    _get_vllm_engine_config,
)
from ray.llm._internal.serve.observability.logging import get_logger
from ray.llm._internal.serve.routing_policies.kv_aware.vllm.prompt_token_forwarding import (  # noqa: E501
    is_text_only_chat,
)

logger = get_logger(__name__)


class TokenizeError(Exception):
    """A client error from vLLM's native prompt renderer.

    Carries the HTTP ``status_code``, ``message`` and error ``type``.
    """

    def __init__(self, message: str, *, status_code: int, type: str):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.type = type


class _NeedsThinkingEffort(Exception):
    """Raised off the event loop when "auto" disabled thinking is unresolved."""


class _InlineRenderer:
    """Delegates to a vLLM renderer but renders in the calling thread.

    Its async render methods return without suspending, so ``OnlineRenderer``
    coroutines built on it run to completion in one ``send()``.
    """

    def __init__(self, renderer):
        self._renderer = renderer

    def __getattr__(self, name):
        return getattr(self._renderer, name)

    async def render_chat_async(self, *args, **kwargs):
        return self._renderer.render_chat(*args, **kwargs)

    async def render_cmpl_async(self, *args, **kwargs):
        return self._renderer.render_cmpl(*args, **kwargs)


def _run_inline(coro):
    """Run a coroutine that never suspends to completion in this thread."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value
    coro.close()
    raise RuntimeError("vLLM's renderer suspended outside the event loop.")


def build_tokenize_request(
    payload: Dict[str, Any],
    *,
    merge_inline_system: bool = True,
    request_path: Optional[str] = None,
) -> Optional[Union[ChatCompletionRequest, CompletionRequest]]:
    """Validate the body using the API path and return a vLLM rendering request.

    Convert Anthropic messages to chat with the engine's ``merge_inline_system``
    setting. The path may include a Serve application prefix.

    Return ``None`` for missing or unsupported paths, invalid bodies, and
    completions without a single string prompt, so routing falls back without
    prompt tokens.

    TODO (jeffreywang): Support multi-prompt tokenization.
    """
    if request_path is None:
        return None
    try:
        if request_path.endswith("/v1/messages"):
            return AnthropicServingMessages.to_chat_completion_request(
                AnthropicMessagesRequest.model_validate(payload),
                merge_inline_system=merge_inline_system,
            )
        if request_path.endswith("/v1/chat/completions"):
            return ChatCompletionRequest.model_validate(payload)
        if request_path.endswith("/v1/completions"):
            request = CompletionRequest.model_validate(payload)
            if not isinstance(request.prompt, str) or request.prompt_embeds is not None:
                return None
            return request
        # Only the generation endpoints above support prompt-token routing here.
        return None
    except (ValidationError, VLLMValidationError) as e:
        # vLLM's request validators can reject sampling params before prompt
        # rendering. Route without tokens so the engine returns its normal
        # client error; failing the router consultation would become a 500.
        logger.warning("Unsupported tokenize request, falling back: %s", e)
        return None


class Tokenizer:
    """Tokenizes requests with vLLM's ``OnlineRenderer``.

    Uses the deployment's frontend args for tokenizer, chat template, and
    trust settings.

    Args:
        llm_config: The deployment's LLM config.
    """

    def __init__(self, llm_config: LLMConfig):
        engine_config = llm_config.get_engine_config()
        _, vllm_config = _get_vllm_engine_config(llm_config, device_type="cpu")
        self._model_config = vllm_config.model_config

        frontend_args = FrontendArgs(**engine_config.frontend_kwargs)
        chat_template = load_chat_template(frontend_args.chat_template)
        self._renderer = OnlineRenderer(
            self._model_config,
            renderer_from_config(vllm_config),
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format=frontend_args.chat_template_content_format,
            trust_request_chat_template=frontend_args.trust_request_chat_template,
            trust_request_mm_kwargs=frontend_args.trust_request_mm_kwargs,
            # Tool handling is a no-op unless the deployment enables tool calling.
            enable_auto_tools=frontend_args.enable_auto_tool_choice,
            exclude_tools_when_tool_choice_none=(
                frontend_args.exclude_tools_when_tool_choice_none
            ),
            tool_parser=frontend_args.tool_call_parser,
            tool_strict_level=frontend_args.tool_strict_level,
            default_chat_template_kwargs=frontend_args.default_chat_template_kwargs,
        )
        # Check the templates the renderer actually uses, including the model's
        # default when no chat-template override is configured.
        self._merge_inline_system = (
            AnthropicServingMessages._should_merge_inline_system(self._renderer)
        )
        self._disabled_thinking_effort = (
            frontend_args.anthropic_disabled_thinking_effort
        )
        # Text-only requests validate, render and tokenize in one hop on the
        # renderer's worker pool, so the event loop awaits a single future.
        self._renderer_impl = self._renderer.renderer
        self._inline_renderer = copy.copy(self._renderer)
        self._inline_renderer.renderer = _InlineRenderer(self._renderer_impl)
        logger.info(
            "In-process pre-routing tokenizer ready for %s",
            self._model_config.model,
        )

    async def tokenize(
        self, payload: Dict[str, Any], *, request_path: Optional[str] = None
    ) -> Optional[List[int]]:
        """Tokenize a request ``payload`` into prompt token IDs.

        Args:
            payload: The request body, already parsed into a dict by ``LLMRouter``.
            request_path: The original API path forwarded by HAProxy, if present.

        Returns:
            The prompt token IDs, or ``None`` for bodies that are not routed on.

        Raises:
            TokenizeError: vLLM's prompt renderer rejected the request.
        """
        if not is_text_only_chat(payload.get("messages", [])):
            # Media needs the renderer's async path, which keeps multimodal
            # processing on its single-worker executor.
            return await self._tokenize(payload, request_path, self._renderer)
        try:
            return await self._tokenize_in_worker(payload, request_path)
        except _NeedsThinkingEffort:
            # Probe templates once on the loop; later requests stay in one hop.
            await self._get_disabled_thinking_effort()
            return await self._tokenize_in_worker(payload, request_path)

    async def _tokenize_in_worker(
        self, payload: Dict[str, Any], request_path: Optional[str]
    ) -> Optional[List[int]]:
        """Validate, render and tokenize a text-only request on a worker."""
        future = asyncio.get_running_loop().run_in_executor(
            self._renderer_impl._executor,
            lambda: _run_inline(
                self._tokenize(payload, request_path, self._inline_renderer)
            ),
        )
        # Bound rendering like vLLM's async path, which replaces the pool so a
        # stuck template thread cannot block later requests.
        timeout = envs.VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT
        if timeout <= 0:
            return await future
        try:
            return await asyncio.wait_for(future, timeout)
        except TimeoutError:
            replace_executor = getattr(self._renderer_impl, "_replace_executor", None)
            if replace_executor is not None:
                replace_executor()
            raise TimeoutError(
                f"Chat template rendering timed out after {timeout}s. This may "
                "indicate a malicious or excessively complex template. Adjust "
                "VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT to override."
            ) from None

    async def _tokenize(
        self,
        payload: Dict[str, Any],
        request_path: Optional[str],
        renderer: OnlineRenderer,
    ) -> Optional[List[int]]:
        request = build_tokenize_request(
            payload,
            merge_inline_system=self._merge_inline_system,
            request_path=request_path,
        )
        if request is None:
            return None

        try:
            if isinstance(request, ChatCompletionRequest):
                # Conversion defaults disabled thinking to "none". Resolve the
                # configured effort before rendering, since it can change tokens.
                if (
                    request_path.endswith("/v1/messages")
                    and request.reasoning_effort == "none"
                ):
                    if (
                        renderer is self._inline_renderer
                        and self._disabled_thinking_effort == "auto"
                    ):
                        raise _NeedsThinkingEffort
                    request.reasoning_effort = (
                        await self._get_disabled_thinking_effort()
                    )
                rendered_inputs = await self._render_chat(request, renderer)
            else:
                rendered_inputs = await self._render_completion(request, renderer)
        except TokenizeError:
            raise
        except (ValueError, VLLMClientError, jinja2.TemplateError) as e:
            # vLLM maps bad inputs and chat-template errors to 400; other
            # exceptions are real bugs and should surface, not degrade routing.
            raise TokenizeError(str(e), status_code=400, type="BadRequestError")

        input_ids: List[int] = []
        for rendered_input in rendered_inputs:
            components = extract_prompt_components(self._model_config, rendered_input)
            if components.token_ids is not None:
                input_ids.extend(components.token_ids)
        return input_ids

    async def _get_disabled_thinking_effort(self):
        """Resolve disabled thinking by rendering local prompts."""
        if self._disabled_thinking_effort != "auto":
            return self._disabled_thinking_effort

        async def probe(effort):
            try:
                (prompt,) = await self._render_chat(
                    ChatCompletionRequest(
                        messages=[{"role": "user", "content": "Hi"}],
                        reasoning_effort=effort,
                    )
                )
                components = extract_prompt_components(self._model_config, prompt)
                return components.token_ids, components.text
            except Exception:
                # Templates may reject an unsupported reasoning effort.
                return None

        # If "none" produces the same prompt as a thinking effort, the template
        # ignores it. Fall back to "low" for ignored or rejected "none".
        none_prompt = await probe("none")
        effort = "none"
        if none_prompt is None:
            effort = "low"
        else:
            for thinking_effort in get_args(AnthropicEffort):
                if await probe(thinking_effort) == none_prompt:
                    effort = "low"
                    break
        # Later requests use this result without rendering probe prompts again.
        self._disabled_thinking_effort = effort
        return effort

    async def _render_chat(
        self,
        request: ChatCompletionRequest,
        renderer: Optional[OnlineRenderer] = None,
    ):
        """Render a chat request to prompt inputs via the engine's own render_chat
        (HF template, Harmony for gpt_oss, Mistral; refuses untrusted templates)."""
        renderer = renderer or self._renderer
        result = await renderer.render_chat(request, skip_mm_cache=True)
        if isinstance(result, ErrorResponse):
            raise TokenizeError(
                result.error.message,
                status_code=result.error.code,
                type=result.error.type,
            )
        _, rendered_inputs = result
        return rendered_inputs

    async def _render_completion(
        self, request: CompletionRequest, renderer: OnlineRenderer
    ):
        return await renderer.preprocess_completion(
            request,
            prompt_input=request.prompt,
            prompt_embeds=None,
            skip_mm_cache=True,
        )
