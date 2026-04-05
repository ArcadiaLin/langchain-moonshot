from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_moonshot.chat_models import ChatMoonshot


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


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


def build_model():
    return ChatMoonshot(
        model="kimi-k2.5",
        api_key=os.getenv("MOONSHOT_API_KEY"),
        api_base=os.getenv("MOONSHOT_API_BASE"),
        temperature=1.0,
    ).bind_tools(TOOLS)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=(
            "You are my AI assistant, please answer my query to the best of "
            "your ability."
        )
    )
    response = build_model().invoke([system_prompt] + list(state["messages"]))
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"


def print_stream(stream) -> None:
    for item in stream:
        message = item["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


def main() -> None:
    graph = StateGraph(AgentState)
    graph.add_node("our_agent", model_call)
    graph.add_node("tools", ToolNode(tools=TOOLS))
    graph.add_edge(START, "our_agent")
    graph.add_conditional_edges(
        "our_agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    graph.add_edge("tools", "our_agent")

    agent = graph.compile()
    input_state = {
        "messages": [
            (
                "user",
                "Add 3 + 4, subtract 5 - 6, multiply 1233 * 4567, also tell me a joke.",
            )
        ]
    }
    print_stream(agent.stream(input_state, stream_mode="values"))


if __name__ == "__main__":
    main()
