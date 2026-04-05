from __future__ import annotations

import argparse
from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from ._common import (
        DEFAULT_TEXT_MODELS,
        DEFAULT_VISION_MODELS,
        image_message_from_path,
        make_model,
    )
except ImportError:
    from _common import (  # type: ignore[no-redef]
        DEFAULT_TEXT_MODELS,
        DEFAULT_VISION_MODELS,
        image_message_from_path,
        make_model,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a smoke matrix across multiple Moonshot models."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_TEXT_MODELS,
        help="Models to test. Defaults to the text model matrix.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with one sentence describing your strongest capability.",
        help="Prompt used for text-only model checks.",
    )
    parser.add_argument(
        "--image-path",
        help="Optional image path. If set, the script also runs the vision matrix.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    all_models = list(args.models)
    if args.image_path:
        for vision_model in DEFAULT_VISION_MODELS:
            if vision_model not in all_models:
                all_models.append(vision_model)

    for model_name in all_models:
        is_k2_5 = model_name == "kimi-k2.5"
        thinking_mode = "on" if is_k2_5 else "default"
        if args.image_path and model_name in DEFAULT_VISION_MODELS:
            prompt = "Describe the image briefly and mention one visual detail."
            content = image_message_from_path(prompt, args.image_path)
        else:
            content = args.prompt

        model = make_model(model_name, thinking_mode=thinking_mode)
        messages = [
            SystemMessage(content="Be concise."),
            HumanMessage(content=content),
        ]

        started_at = perf_counter()
        try:
            response = model.invoke(messages)
            elapsed = perf_counter() - started_at
            text = response.text.replace("\n", " ").strip()
            preview = text[:120]
            print(f"[PASS] {model_name} {elapsed:.2f}s {preview}")
        except Exception as exc:  # noqa: BLE001
            elapsed = perf_counter() - started_at
            print(f"[FAIL] {model_name} {elapsed:.2f}s {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
