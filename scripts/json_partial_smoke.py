from __future__ import annotations

import argparse
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run JSON mode and partial continuation smoke tests."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--json-prompt",
        default=(
            "Return a JSON object with keys summary and tags. "
            "summary should be one sentence. tags should be a short array of strings."
        ),
        help="Prompt used for the JSON mode check.",
    )
    parser.add_argument(
        "--partial-prompt",
        default=(
            "Write a short paragraph about why agent integrations need smoke tests."
        ),
        help="Prompt used for the partial continuation check.",
    )
    parser.add_argument(
        "--initial-max-completion-tokens",
        type=int,
        default=96,
        help="Small token budget for the first partial step.",
    )
    parser.add_argument(
        "--continuation-max-completion-tokens",
        type=int,
        default=256,
        help="Token budget for each continuation step.",
    )
    return parser


def run_json_mode(args: argparse.Namespace) -> None:
    model = make_model(args.model, thinking_mode=args.thinking)
    messages = [
        SystemMessage(
            content=(
                "You must output a valid JSON object with keys summary and tags. "
                "summary must be a string. tags must be an array of strings."
            )
        ),
        HumanMessage(content=args.json_prompt),
    ]

    response = model.invoke(
        messages,
        response_format={"type": "json_object"},
    )
    print("json mode raw content:")
    print(response.content)
    print("\njson mode parsed object:")
    print(json.loads(response.content))
    print("\njson mode message:")
    print_message(response)


def run_partial_continuation(args: argparse.Namespace) -> None:
    model = make_model(args.model, thinking_mode=args.thinking)
    base_messages = [
        SystemMessage(content="You are a concise assistant."),
        HumanMessage(content=args.partial_prompt),
    ]

    first = model.invoke(
        base_messages,
        max_completion_tokens=args.initial_max_completion_tokens,
    )
    print("\npartial step 1:")
    print_message(first)

    finish_reason = first.response_metadata.get("finish_reason")
    print(f"\npartial step 1 finish_reason: {finish_reason!r}")
    if finish_reason != "length":
        print(
            "\npartial continuation skipped because the first response did not "
            f"truncate. finish_reason={finish_reason!r}"
        )
        return

    full_content = first.text
    latest_reasoning = first.additional_kwargs.get("reasoning_content")
    step = 2

    while finish_reason == "length":
        continuation_messages = [
            *base_messages,
            AIMessage(
                content=full_content,
                additional_kwargs={
                    "partial": True,
                    "reasoning_content": latest_reasoning,
                },
            ),
        ]
        response = model.invoke(
            continuation_messages,
            max_completion_tokens=args.continuation_max_completion_tokens,
        )

        print(f"\npartial step {step} continuation message:")
        print_message(response)

        finish_reason = response.response_metadata.get("finish_reason")
        print(f"\npartial step {step} finish_reason: {finish_reason!r}")

        full_content += response.text
        latest_reasoning = response.additional_kwargs.get(
            "reasoning_content",
            latest_reasoning,
        )
        step += 1

    print("\npartial reconstructed full content:")
    print(full_content)


def main() -> None:
    args = build_parser().parse_args()
    print(f"model={args.model}")
    print("\n=== JSON mode ===")
    run_json_mode(args)
    print("\n=== Partial continuation ===")
    run_partial_continuation(args)


if __name__ == "__main__":
    main()
