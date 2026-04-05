from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr

from langchain_moonshot import ChatMoonshot

DEFAULT_API_BASE = "https://api.moonshot.cn/v1"

pytestmark = pytest.mark.skipif(
    not (
        os.getenv("RUN_MOONSHOT_INTEGRATION") == "1"
        and os.getenv("MOONSHOT_API_KEY")
    ),
    reason=(
        "Set RUN_MOONSHOT_INTEGRATION=1 and MOONSHOT_API_KEY to run live "
        "Moonshot integration tests."
    ),
)


def _make_model(
    *,
    model: str = "kimi-k2.5",
    thinking: bool | None = True,
    stream_usage: bool = False,
    **kwargs: Any,
) -> ChatMoonshot:
    temperature = kwargs.pop("temperature", None)
    if model == "kimi-k2.5" and temperature is None:
        temperature = 1.0 if thinking is not False else 0.6

    return ChatMoonshot(
        model=model,
        api_key=SecretStr(os.environ["MOONSHOT_API_KEY"]),
        base_url=os.getenv("MOONSHOT_API_BASE", DEFAULT_API_BASE),
        thinking=thinking,
        temperature=temperature,
        stream_usage=stream_usage,
        **kwargs,
    )


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


TOOLS = [add, subtract, multiply]
TOOL_BY_NAME = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
}


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def test_chat_moonshot_invoke_smoke() -> None:
    model = _make_model()

    message = model.invoke("Reply with the single word pong.")

    assert isinstance(message.content, str)
    assert message.content
    assert message.response_metadata["model_provider"] == "moonshot"


def test_chat_moonshot_stream_smoke() -> None:
    model = _make_model(stream_usage=True)
    chunks = list(
        model.stream(
            [
                SystemMessage(content="You are a concise assistant."),
                HumanMessage(content="Give two short reasons why streaming matters."),
            ]
        )
    )

    assert chunks

    full = chunks[0]
    for chunk in chunks[1:]:
        full += chunk

    assert full.text.strip()
    assert full.response_metadata["model_provider"] == "moonshot"
    if full.usage_metadata is not None:
        assert full.usage_metadata["total_tokens"] > 0


def test_chat_moonshot_tool_loop_smoke() -> None:
    model = _make_model(thinking=False).bind_tools(TOOLS)
    messages: list[BaseMessage] = [
        SystemMessage(content="Use the provided math tools before answering."),
        HumanMessage(
            content=(
                "Use the tools to add 17 and 25, subtract 30 from 45, "
                "and multiply 12 by 13. Then summarize the results briefly."
            )
        ),
    ]

    tool_names_seen: set[str] = set()
    final_message: AIMessage | None = None

    for _ in range(4):
        response = model.invoke(messages)
        assert isinstance(response, AIMessage)
        messages.append(response)

        if not response.tool_calls:
            final_message = response
            break

        for tool_call in response.tool_calls:
            tool_names_seen.add(tool_call["name"])
            tool_result = TOOL_BY_NAME[tool_call["name"]].invoke(tool_call["args"])
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )

    assert tool_names_seen
    assert final_message is not None
    assert isinstance(final_message.content, str)
    assert final_message.content


def test_chat_moonshot_langgraph_agent_smoke() -> None:
    model = _make_model(thinking=False).bind_tools(TOOLS)

    def model_call(state: AgentState) -> AgentState:
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a careful assistant. Use the provided tools when "
                        "they help answer the user."
                    )
                ),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", model_call)
    graph.add_node("tools", ToolNode(tools=TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    graph.add_edge("tools", "agent")

    result = graph.compile().invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Use the available tools to add 3 and 4, subtract 5 and 6, "
                        "and multiply 12 and 13. Then tell me one short joke."
                    )
                )
            ]
        }
    )

    assert any(isinstance(message, ToolMessage) for message in result["messages"])
    final_message = result["messages"][-1]
    assert isinstance(final_message, AIMessage)
    assert isinstance(final_message.content, str)
    assert final_message.content
