from __future__ import annotations

import argparse

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from ._common import (
        add_common_arguments,
        image_message_from_path,
        make_model,
        print_message,
        stream_and_collect,
    )
except ImportError:
    from _common import (  # type: ignore[no-redef]
        add_common_arguments,
        image_message_from_path,
        make_model,
        print_message,
        stream_and_collect,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream a Moonshot chat response.")
    add_common_arguments(parser)
    parser.add_argument(
        "--prompt",
        default="Explain in three short bullet points why streaming output matters.",
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--system",
        default="You are a concise assistant.",
        help="System prompt.",
    )
    parser.add_argument(
        "--image-path",
        help="Optional local image path for multimodal streaming smoke tests.",
    )
    parser.add_argument(
        "--stream-usage",
        action="store_true",
        help="Request streaming usage metadata.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = make_model(
        args.model,
        thinking_mode=args.thinking,
        stream_usage=args.stream_usage,
    )

    if args.image_path:
        messages = [
            SystemMessage(content=args.system),
            HumanMessage(content=image_message_from_path(args.prompt, args.image_path)),
        ]
    else:
        messages = [
            SystemMessage(content=args.system),
            HumanMessage(content=args.prompt),
        ]

    print(f"model={args.model}")
    print("stream output:")
    full = stream_and_collect(model, messages)
    if full is not None:
        print("\nfinal assembled message:")
        print_message(full)


if __name__ == "__main__":
    main()
