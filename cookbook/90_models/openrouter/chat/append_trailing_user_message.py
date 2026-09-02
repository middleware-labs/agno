"""
OpenRouter Append Trailing User Message
=======================================

Claude 4.6+ does not support assistant message prefill and returns a 400 when a request
ends with an assistant turn (which happens during reasoning). `append_trailing_user_message`
appends a trailing user turn in that case.

On OpenRouter the model id is often not the model that answers, so the flag is auto-enabled
whenever any model that could serve the request rejects prefill:

- `openrouter/auto` and `@preset/...` ids, because the router picks the model per request
- a concrete Claude 4.6+ id, e.g. `anthropic/claude-sonnet-4-6`
- any `models` fallback entry that rejects prefill

Use `trailing_user_message_content` to customise the appended text (defaults to "continue").
"""

from agno.agent import Agent
from agno.models.openrouter import OpenRouter

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------

# The Auto Router may pick a Claude 4.6+ model, so the guard is enabled automatically.
auto_router_agent = Agent(
    model=OpenRouter(id="openrouter/auto"),
    reasoning=True,
    markdown=True,
)

# A non-Claude primary with a prefill-unsupported fallback also enables the guard.
fallback_agent = Agent(
    model=OpenRouter(
        id="openai/gpt-4o",
        models=["anthropic/claude-sonnet-4-6"],
    ),
    reasoning=True,
    markdown=True,
)

# Set the flag explicitly to override auto-detection, and customise the injected text.
explicit_agent = Agent(
    model=OpenRouter(
        id="openrouter/auto",
        append_trailing_user_message=True,
        trailing_user_message_content="continue",
    ),
    reasoning=True,
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    auto_router_agent.print_response("What is 15 + 27?")
    fallback_agent.print_response("What is 15 + 27?")
    explicit_agent.print_response("What is 15 + 27?")
