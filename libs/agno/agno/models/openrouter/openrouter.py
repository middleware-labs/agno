from dataclasses import dataclass
from os import getenv
from typing import Any, Dict, List, Optional, Type, Union

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

from agno.exceptions import ModelAuthenticationError
from agno.media import Image
from agno.models.message import Message
from agno.models.openai.like import OpenAILike
from agno.models.response import ModelResponse
from agno.run.agent import RunOutput
from agno.utils.models.claude import supports_prefill

# Model ids that name a router rather than a model: the upstream model is chosen by
# OpenRouter per request, so the id says nothing about whether prefill is supported.
_OPAQUE_ROUTED_IDS = frozenset({"openrouter/auto"})
_OPAQUE_ROUTED_ID_PREFIXES = ("@preset/",)


@dataclass
class OpenRouter(OpenAILike):
    """
    A class for using models hosted on OpenRouter.

    Attributes:
        id (str): The model id. Defaults to "gpt-5.4-mini".
        name (str): The model name. Defaults to "OpenRouter".
        provider (str): The provider name. Defaults to "OpenRouter".
        api_key (Optional[str]): The API key.
        base_url (str): The base URL. Defaults to "https://openrouter.ai/api/v1".
        max_tokens (int): The maximum number of tokens. Defaults to 1024.
        fallback_models (Optional[List[str]]): List of fallback model IDs to use if the primary model
            fails due to rate limits, timeouts, or unavailability. OpenRouter will automatically try
            these models in order. Example: ["anthropic/claude-sonnet-4", "deepseek/deepseek-r1"]
        append_trailing_user_message (Optional[bool]): Inherited from ``OpenAILike``. Defaults to
            None, which auto-detects from every model that can serve the request, not just ``id``
            (see ``__post_init__``).
    """

    id: str = "gpt-5.4-mini"
    name: str = "OpenRouter"
    provider: str = "OpenRouter"

    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 1024
    models: Optional[List[str]] = None  # Dynamic model routing https://openrouter.ai/docs/features/model-routing

    def __post_init__(self):
        # Read the flag before OpenAILike resolves None into a concrete bool, so an
        # explicit caller value still wins over the router-aware default below.
        prefill_guard_unset = self.append_trailing_user_message is None

        super().__post_init__()

        # OpenAILike auto-detects the prefill guard from `id` alone. On OpenRouter the id
        # is often not the model that answers: the Auto Router and presets resolve upstream,
        # and `models` adds fallbacks. So the inherited check clears the guard while the
        # request can still land on Claude 4.6+, which rejects a trailing assistant message
        # with a 400.
        if prefill_guard_unset and self._may_route_to_prefill_unsupported_model():
            self.append_trailing_user_message = True

    def _may_route_to_prefill_unsupported_model(self) -> bool:
        """Whether OpenRouter may serve this request from a model that rejects assistant prefill."""
        if self.id in _OPAQUE_ROUTED_IDS or self.id.startswith(_OPAQUE_ROUTED_ID_PREFIXES):
            # The upstream model is unknowable here, so assume the stricter contract.
            return True

        return any(not supports_prefill(model_id) for model_id in self.models or [])

    def _get_client_params(self) -> Dict[str, Any]:
        """
        Returns client parameters for API requests, checking for OPENROUTER_API_KEY.

        Returns:
            Dict[str, Any]: A dictionary of client parameters for API requests.
        """
        # Fetch API key from env if not already set
        if not self.api_key:
            self.api_key = getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ModelAuthenticationError(
                    message="OPENROUTER_API_KEY not set. Please set the OPENROUTER_API_KEY environment variable.",
                    model_name=self.name,
                )

        return super()._get_client_params()

    def get_request_params(
        self,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_response: Optional[RunOutput] = None,
    ) -> Dict[str, Any]:
        """
        Returns keyword arguments for API requests, including fallback models configuration.

        Returns:
            Dict[str, Any]: A dictionary of keyword arguments for API requests.
        """
        # Get base request params from parent class
        request_params = super().get_request_params(
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            run_response=run_response,
        )

        # Add fallback models to extra_body if specified
        if self.models:
            # Get existing extra_body or create new dict
            extra_body = request_params.get("extra_body") or {}

            # Merge fallback models into extra_body
            extra_body["models"] = self.models

            # Update request params
            request_params["extra_body"] = extra_body

        return request_params

    def _format_message(self, message: Message, compress_tool_results: bool = False) -> Dict[str, Any]:
        message_dict = super()._format_message(message, compress_tool_results)

        if message.role == "assistant" and message.provider_data:
            if message.provider_data.get("reasoning_details"):
                message_dict["reasoning_details"] = message.provider_data["reasoning_details"]

        return message_dict

    def _add_openrouter_images(self, extra: Optional[Any], model_response: ModelResponse) -> None:
        """Append generated images from OpenRouter's ``model_extra["images"]`` to model_response.

        OpenRouter returns generated images as data URLs under the assistant message's
        (and streamed delta's) ``images`` field, e.g.
        ``{"image_url": {"url": "data:image/png;base64,..."}}``.
        """
        if not isinstance(extra, dict) or not isinstance(extra.get("images"), list):
            return
        for item in extra["images"]:
            image_url = item.get("image_url") if isinstance(item, dict) else None
            url = image_url.get("url") if isinstance(image_url, dict) else None
            # Only data URLs are expected here. Skip remote URLs rather than fetch a
            # provider-controlled address server-side (SSRF risk via Image.get_content_bytes).
            if not url or not url.startswith("data:"):
                continue
            header, _, payload = url.partition(",")
            mime_type = header[len("data:") :].split(";")[0] or None
            image = Image.from_base64(payload, mime_type=mime_type)
            if model_response.images is None:
                model_response.images = []
            model_response.images.append(image)

    def _parse_provider_response(
        self,
        response: ChatCompletion,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> ModelResponse:
        model_response = super()._parse_provider_response(response, response_format)

        if response.choices and len(response.choices) > 0:
            response_message = response.choices[0].message
            if hasattr(response_message, "reasoning_details") and response_message.reasoning_details:
                if model_response.provider_data is None:
                    model_response.provider_data = {}
                model_response.provider_data["reasoning_details"] = response_message.reasoning_details
            elif hasattr(response_message, "model_extra"):
                extra = getattr(response_message, "model_extra", None)
                if extra and isinstance(extra, dict) and extra.get("reasoning_details"):
                    if model_response.provider_data is None:
                        model_response.provider_data = {}
                    model_response.provider_data["reasoning_details"] = extra["reasoning_details"]

            # Generated images arrive under model_extra["images"] (buffered response)
            self._add_openrouter_images(getattr(response_message, "model_extra", None), model_response)

        return model_response

    def _parse_provider_response_delta(self, response_delta: ChatCompletionChunk) -> ModelResponse:
        model_response = super()._parse_provider_response_delta(response_delta)

        if response_delta.choices and len(response_delta.choices) > 0:
            choice_delta = response_delta.choices[0].delta
            if hasattr(choice_delta, "reasoning_details") and choice_delta.reasoning_details:
                if model_response.provider_data is None:
                    model_response.provider_data = {}
                model_response.provider_data["reasoning_details"] = choice_delta.reasoning_details

            # Streamed generated images arrive under delta.images (same shape as buffered)
            self._add_openrouter_images(getattr(choice_delta, "model_extra", None), model_response)

        return model_response
