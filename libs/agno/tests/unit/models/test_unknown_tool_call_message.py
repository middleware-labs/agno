"""Tests for the tool result returned when a model calls a tool that does not exist.

Team leaders sometimes call a member id (e.g. "code-it-up-agent") as if it were a
tool. The error result must tell the model how to delegate correctly so it can
recover on the next turn instead of dropping the step.
"""

import os
from typing import Dict, List

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")

from agno.models.message import Message
from agno.models.openai.chat import OpenAIChat
from agno.tools.function import Function


def _delegate_task_to_member(member_id: str, task: str) -> str:
    return f"{member_id}: {task}"


def _update_session_state(session_state_updates: dict) -> str:
    return "ok"


@pytest.fixture
def model():
    return OpenAIChat(id="gpt-4o-mini")


def _tool_call(call_id: str, name: str, arguments: str) -> Dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _run(model, functions: Dict[str, Function], tool_calls: List[Dict]) -> List[Message]:
    messages: List[Message] = []
    model.get_function_calls_to_run(
        assistant_message=Message(role="assistant", tool_calls=tool_calls),
        messages=messages,
        functions=functions,
    )
    return messages


def test_member_id_called_as_tool_points_to_delegate_task_to_member(model):
    functions = {
        "delegate_task_to_member": Function.from_callable(_delegate_task_to_member, name="delegate_task_to_member"),
        "update_session_state": Function.from_callable(_update_session_state, name="update_session_state"),
    }

    messages = _run(model, functions, [_tool_call("call_1", "code-it-up-agent", '{"task": "read repo"}')])

    assert len(messages) == 1
    content = messages[0].content
    assert messages[0].tool_call_id == "call_1"
    assert messages[0].tool_name == "code-it-up-agent"
    assert "`code-it-up-agent` is not a tool" in content
    assert 'delegate_task_to_member(member_id="code-it-up-agent", task=...)' in content
    assert "did not reach any team member" in content


def test_unknown_tool_without_delegation_keeps_generic_error(model):
    functions = {
        "update_session_state": Function.from_callable(_update_session_state, name="update_session_state"),
    }

    messages = _run(model, functions, [_tool_call("call_1", "missing_tool", "{}")])

    assert len(messages) == 1
    assert messages[0].content == "Error: The requested tool does not exist or is not available."


def test_known_tool_in_same_turn_still_runs(model):
    functions = {
        "delegate_task_to_member": Function.from_callable(_delegate_task_to_member, name="delegate_task_to_member"),
    }

    messages: List[Message] = []
    function_calls = model.get_function_calls_to_run(
        assistant_message=Message(
            role="assistant",
            tool_calls=[
                _tool_call("call_1", "delegate_task_to_member", '{"member_id": "planner-agent", "task": "plan"}'),
                _tool_call("call_2", "code-it-up-agent", '{"task": "read repo"}'),
            ],
        ),
        messages=messages,
        functions=functions,
    )

    assert [fc.function.name for fc in function_calls] == ["delegate_task_to_member"]
    assert len(messages) == 1
    assert messages[0].tool_call_id == "call_2"
