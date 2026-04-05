from __future__ import annotations

import argparse
from collections.abc import Callable

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

try:
    from ._common import (
        add_common_arguments,
        make_model,
        print_message,
    )
except ImportError:
    from _common import (  # type: ignore[no-redef]
        add_common_arguments,
        make_model,
        print_message,
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
TOOL_BY_NAME: dict[str, Callable[..., int]] = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a manual tool-calling loop with ChatMoonshot."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--prompt",
        default=(
            "Add 17 and 25, subtract 30 from 45, multiply 12 by 13, "
            "then explain the results briefly."
        ),
        help="Prompt to send to the model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = make_model(args.model, thinking_mode=args.thinking).bind_tools(TOOLS)

    messages: list[BaseMessage | tuple[str, str]] = [
        SystemMessage(content="You may use the provided math tools when needed."),
        ("user", args.prompt),
    ]

    print(f"model={args.model}")
    while True:
        response = model.invoke(messages)
        assert isinstance(response, AIMessage)

        print("\nassistant:")
        print_message(response)
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_result = TOOL_BY_NAME[tool_name].invoke(tool_call["args"])
            tool_message = ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            )
            print("\ntool result:")
            print(tool_message.content)
            messages.append(tool_message)


if __name__ == "__main__":
    main()
