from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agno.models.base import Model
from agno.models.message import Message
from agno.models.openai.chat import OpenAIChat
from agno.utils.log import log_info
from agno.utils.models.claude import supports_prefill


@dataclass
class OpenAILike(OpenAIChat):
    """
    A class for to interact with any provider using the OpenAI API schema.

    Args:
        id (str): The id of the OpenAI model to use. Defaults to "not-provided".
        name (str): The name of the OpenAI model to use. Defaults to "OpenAILike".
        api_key (Optional[str]): The API key to use. Defaults to "not-provided".
        append_trailing_user_message (Optional[bool]): Append a trailing user turn when the
            conversation ends with an assistant message. Defaults to None (auto-detect from ``id``).
        trailing_user_message_content (str): Text of the injected trailing user message.
    """

    id: str = "not-provided"
    name: str = "OpenAILike"
    api_key: Optional[str] = "not-provided"

    # Claude 4.6+ rejects a trailing assistant message (assistant prefill) with a 400.
    # OpenAI-compatible gateways can forward to those models, so the same guard is needed
    # here. None auto-detects from ``id``; gateways and routers that hide the upstream
    # model behind an opaque id must set this to True, since the id reveals nothing.
    append_trailing_user_message: Optional[bool] = None
    trailing_user_message_content: str = "continue"

    default_role_map = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
    }

    def __post_init__(self):
        super().__post_init__()

        if self.append_trailing_user_message is None:
            self.append_trailing_user_message = not supports_prefill(self.id)

    def get_provider(self) -> str:
        return Model.get_provider(self)

    def _format_all_messages(
        self, messages: List[Message], compress_tool_results: bool = False
    ) -> List[Dict[str, Any]]:
        formatted_messages = super()._format_all_messages(messages, compress_tool_results)

        role_map = self.role_map or self.default_role_map
        if (
            self.append_trailing_user_message
            and formatted_messages
            and formatted_messages[-1].get("role") == role_map.get("assistant", "assistant")
        ):
            log_info("Appending trailing user message because this model does not support assistant message prefill")
            formatted_messages.append(
                {"role": role_map.get("user", "user"), "content": self.trailing_user_message_content}
            )

        return formatted_messages
