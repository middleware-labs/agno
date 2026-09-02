"""Tests for append_trailing_user_message auto-detection on OpenRouter.

OpenAILike auto-detects the prefill guard from `id`, but an OpenRouter id is often not
the model that answers: the Auto Router and preset slugs resolve upstream, and `models`
adds fallbacks. OpenRouter must therefore enable the guard whenever any model that could
serve the request rejects assistant prefill (Claude 4.6+ returns a 400).
"""

from agno.models.message import Message
from agno.models.openrouter import OpenRouter

ENDS_WITH_ASSISTANT = [
    Message(role="user", content="Classify this ticket: checkout is broken"),
    Message(role="assistant", content='{"priority":'),
]

ENDS_WITH_USER = [
    Message(role="user", content="What is 2+2?"),
]


class TestAutoDetectionFromId:
    """The guard is derived from the id when the id names a concrete model."""

    def test_concrete_claude_46_auto_enabled(self):
        assert OpenRouter(id="anthropic/claude-sonnet-4-6").append_trailing_user_message is True

    def test_concrete_claude_45_auto_disabled(self):
        assert OpenRouter(id="anthropic/claude-sonnet-4-5").append_trailing_user_message is False

    def test_non_claude_auto_disabled(self):
        assert OpenRouter(id="openai/gpt-4o").append_trailing_user_message is False


class TestAutoDetectionFromRouterId:
    """Router and preset ids hide the upstream model, so the guard defaults on."""

    def test_auto_router_enabled(self):
        assert OpenRouter(id="openrouter/auto").append_trailing_user_message is True

    def test_preset_slug_enabled(self):
        assert OpenRouter(id="@preset/my-ops-preset").append_trailing_user_message is True


class TestAutoDetectionFromFallbackModels:
    """`models` routing means a non-Claude id can still be served by Claude 4.6+."""

    def test_prefill_unsupported_fallback_enables_guard(self):
        model = OpenRouter(id="openai/gpt-4o", models=["openai/gpt-4o", "anthropic/claude-sonnet-4-6"])
        assert model.append_trailing_user_message is True

    def test_all_prefill_supported_fallbacks_leave_guard_off(self):
        model = OpenRouter(id="openai/gpt-4o", models=["openai/gpt-4o", "anthropic/claude-sonnet-4-5"])
        assert model.append_trailing_user_message is False


class TestExplicitOverride:
    """An explicit flag always wins over auto-detection."""

    def test_disabled_on_auto_router(self):
        model = OpenRouter(id="openrouter/auto", append_trailing_user_message=False)
        assert model.append_trailing_user_message is False

    def test_enabled_on_non_claude_id(self):
        model = OpenRouter(id="openai/gpt-4o", append_trailing_user_message=True)
        assert model.append_trailing_user_message is True


class TestFormatAllMessages:
    """The resolved guard actually appends the trailing user turn."""

    def test_appends_when_ends_with_assistant(self):
        formatted = OpenRouter(id="openrouter/auto")._format_all_messages(ENDS_WITH_ASSISTANT)
        assert formatted[-1]["role"] == "user"
        assert formatted[-1]["content"] == "continue"

    def test_no_append_when_ends_with_user(self):
        formatted = OpenRouter(id="openrouter/auto")._format_all_messages(ENDS_WITH_USER)
        assert formatted[-1]["role"] == "user"
        assert len(formatted) == 1

    def test_no_append_for_prefill_supported_id(self):
        formatted = OpenRouter(id="anthropic/claude-sonnet-4-5")._format_all_messages(ENDS_WITH_ASSISTANT)
        assert formatted[-1]["role"] == "assistant"

    def test_custom_trailing_content(self):
        model = OpenRouter(id="openrouter/auto", trailing_user_message_content="go on")
        formatted = model._format_all_messages(ENDS_WITH_ASSISTANT)
        assert formatted[-1]["role"] == "user"
        assert formatted[-1]["content"] == "go on"
