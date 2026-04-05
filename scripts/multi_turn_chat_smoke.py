from __future__ import annotations

import argparse

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

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

DEFAULT_TURNS = [
    "My name is Ada. I work on agents.",
    "Summarize what you learned about me in one sentence.",
    "Now suggest two good smoke tests for an agent integration package.",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a multi-turn conversation against ChatMoonshot."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--system",
        default="You are a helpful assistant and must remember prior turns.",
        help="System prompt.",
    )
    parser.add_argument(
        "--user",
        action="append",
        dest="user_turns",
        help="User turn. Repeat to add multiple turns.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = make_model(args.model, thinking_mode=args.thinking)

    history: list[BaseMessage] = [SystemMessage(content=args.system)]
    user_turns = args.user_turns or DEFAULT_TURNS

    print(f"model={args.model}")
    for index, user_turn in enumerate(user_turns, start=1):
        print(f"\nturn {index} user:")
        print(user_turn)

        history.append(HumanMessage(content=user_turn))
        response = model.invoke(history)
        assert isinstance(response, AIMessage)

        print(f"\nturn {index} assistant:")
        print_message(response)
        history.append(response)


if __name__ == "__main__":
    main()
