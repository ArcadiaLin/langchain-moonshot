from __future__ import annotations

import argparse

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from ._common import (
        DEFAULT_VISION_MODELS,
        image_message_from_path,
        make_model,
        print_message,
    )
except ImportError:
    from _common import (  # type: ignore[no-redef]
        DEFAULT_VISION_MODELS,
        image_message_from_path,
        make_model,
        print_message,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Moonshot vision smoke test against one or more models."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_VISION_MODELS,
        help="Vision-capable models to test.",
    )
    parser.add_argument(
        "--image-path",
        required=True,
        help="Local image path to send as a multimodal input.",
    )
    parser.add_argument(
        "--prompt",
        default="Describe the image and mention one concrete visual detail.",
        help="Prompt paired with the image.",
    )
    parser.add_argument(
        "--system",
        default="You are a concise vision assistant.",
        help="System prompt.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming mode for the first model only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    for index, model_name in enumerate(args.models):
        model = make_model(
            model_name,
            thinking_mode="on" if model_name == "kimi-k2.5" else "default",
            stream_usage=args.stream and index == 0,
        )
        messages = [
            SystemMessage(content=args.system),
            HumanMessage(content=image_message_from_path(args.prompt, args.image_path)),
        ]

        print(f"\nmodel={model_name}")
        if args.stream and index == 0:
            print("stream output:")
            full = None
            for chunk in model.stream(messages):
                if full is None:
                    full = chunk
                else:
                    full += chunk
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print()
            if full is not None:
                print("\nfinal assembled message:")
                print_message(full)
            continue

        response = model.invoke(messages)
        print_message(response)


if __name__ == "__main__":
    main()
