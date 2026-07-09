from typing import Any, Dict, List, Optional, Union

import jinja2
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.openai.cli_args import FrontendArgs
from vllm.renderers import renderer_from_config
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.renderers.online_renderer import OnlineRenderer

from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from ray.llm._internal.serve.core.configs.openai_api_models import (
    TokenizeChatRequest,
    TokenizeCompletionRequest,
)
from ray.llm._internal.serve.engines.vllm.vllm_engine import (
    _get_vllm_engine_config,
)
from ray.llm._internal.serve.observability.logging import get_logger

logger = get_logger(__name__)


class TokenizeError(Exception):
    """The request was rejected the same way vLLM's native ASGI route
    ``/tokenize`` would reject it.

    Carries the HTTP ``status_code``, ``message`` and error ``type``.
    """

    def __init__(self, message: str, *, status_code: int, type: str):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.type = type


def build_tokenize_request(
    payload: Dict[str, Any]
) -> Optional[Union[TokenizeChatRequest, TokenizeCompletionRequest]]:
    """Build the Tokenize* request for ``payload``, forwarding only the fields
    the engine renders the prompt from so routing ids match the prefill tokens.

    Returns ``None`` (caller falls back to token-less routing) for a body with
    no single string prompt, e.g. a batch ``prompt`` list, since KV-aware
    routing scores one request on one token sequence.

    TODO (jeffreywang): Support multi-prompt tokenization.
    """
    try:
        if "messages" in payload:
            return TokenizeChatRequest.model_validate(
                {
                    k: v
                    for k, v in payload.items()
                    if k in TokenizeChatRequest.model_fields
                }
            )
        if "prompt" in payload:
            if not isinstance(payload["prompt"], str):
                return None
            return TokenizeCompletionRequest.model_validate(
                {
                    k: v
                    for k, v in payload.items()
                    if k in TokenizeCompletionRequest.model_fields
                }
            )
        # Unreachable: LLMRouter only routes bodies with messages or a prompt.
        logger.warning(
            "Tokenizer got a payload with neither messages nor prompt; "
            "falling back to token-less routing."
        )
        return None
    except Exception as e:
        logger.warning("Unsupported tokenize request, falling back: %s", e)
        return None


class Tokenizer:
    """Tokenizes requests with vLLM's ``OnlineRenderer``.

    Configured from the deployment's frontend args so the tokenizer, chat
    template, and trust policy match the engine's.

    Args:
        llm_config: The deployment's LLM config.
    """

    def __init__(self, llm_config: LLMConfig):
        engine_config = llm_config.get_engine_config()
        _, vllm_config = _get_vllm_engine_config(llm_config)
        self._model_config = vllm_config.model_config

        frontend_args = FrontendArgs(**engine_config.frontend_kwargs)
        self._renderer = OnlineRenderer(
            self._model_config,
            renderer_from_config(vllm_config),
            request_logger=None,
            chat_template=load_chat_template(frontend_args.chat_template),
            chat_template_content_format=frontend_args.chat_template_content_format,
            trust_request_chat_template=frontend_args.trust_request_chat_template,
            default_chat_template_kwargs=frontend_args.default_chat_template_kwargs,
        )
        logger.info(
            "In-process pre-routing tokenizer ready for %s",
            self._model_config.model,
        )

    async def tokenize(self, payload: Dict[str, Any]) -> Optional[List[int]]:
        """Tokenize a request ``payload`` into prompt token IDs.

        Args:
            payload: The request body, already parsed into a dict by ``LLMRouter``.

        Returns:
            The prompt token IDs, or ``None`` for bodies that are not routed on.

        Raises:
            TokenizeError: The ``/tokenize`` endpoint rejected the request.
        """
        request = build_tokenize_request(payload)
        if request is None:
            return None

        try:
            if isinstance(request, TokenizeChatRequest):
                engine_inputs = await self._render_chat(request)
            else:
                engine_inputs = await self._renderer.preprocess_completion(
                    request,
                    prompt_input=request.prompt,
                    prompt_embeds=None,
                    skip_mm_cache=True,
                )
        except TokenizeError:
            raise
        except (ValueError, jinja2.TemplateError) as e:
            # /tokenize maps bad inputs and chat-template errors to 400; other
            # exceptions are real bugs and should surface, not degrade routing.
            raise TokenizeError(str(e), status_code=400, type="BadRequestError")

        input_ids: List[int] = []
        for engine_input in engine_inputs:
            components = extract_prompt_components(self._model_config, engine_input)
            if components.token_ids is not None:
                input_ids.extend(components.token_ids)
        return input_ids

    async def _render_chat(self, request: TokenizeChatRequest):
        # Refuse a request-supplied chat template unless the deployment opted in.
        error = self._renderer.validate_chat_template(
            request_chat_template=request.chat_template,
            chat_template_kwargs=request.chat_template_kwargs,
            trust_request_chat_template=self._renderer.trust_request_chat_template,
        )
        if error is not None:
            raise TokenizeError(
                error.error.message,
                status_code=error.error.code,
                type=error.error.type,
            )

        tool_dicts = (
            None
            if request.tools is None
            else [tool.model_dump() for tool in request.tools]
        )
        _, engine_inputs = await self._renderer.preprocess_chat(
            request,
            request.messages,
            default_template=self._renderer.chat_template,
            default_template_content_format=self._renderer.chat_template_content_format,
            default_template_kwargs=self._renderer.default_chat_template_kwargs,
            tool_dicts=tool_dicts,
            skip_mm_cache=True,
        )
        return engine_inputs
