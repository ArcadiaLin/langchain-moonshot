from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from langchain_moonshot import ChatMoonshot

DEFAULT_API_BASE = "https://api.moonshot.cn/v1"
DEFAULT_TEXT_MODELS = [
    "kimi-k2.5",
    "kimi-k2-thinking",
    "kimi-k2-thinking-turbo",
    "moonshot-v1-8k",
    "moonshot-v1-32k",
    "moonshot-v1-128k",
]
DEFAULT_VISION_MODELS = [
    "kimi-k2.5",
    "moonshot-v1-8k-vision-preview",
    "moonshot-v1-32k-vision-preview",
    "moonshot-v1-128k-vision-preview",
]


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required.")
    return value


def make_model(
    model_name: str,
    *,
    thinking_mode: str = "default",
    stream_usage: bool = False,
    **kwargs: Any,
) -> ChatMoonshot:
    thinking: bool | None
    if thinking_mode == "on":
        thinking = True
    elif thinking_mode == "off":
        thinking = False
    else:
        thinking = None

    temperature = kwargs.pop("temperature", None)
    if model_name == "kimi-k2.5" and temperature is None:
        temperature = 1.0 if thinking is not False else 0.6

    return ChatMoonshot(
        model=model_name,
        api_key=require_env("MOONSHOT_API_KEY"),
        base_url=os.getenv("MOONSHOT_API_BASE", DEFAULT_API_BASE),
        thinking=thinking,
        temperature=temperature,
        stream_usage=stream_usage,
        **kwargs,
    )


def build_text_messages(
    system_prompt: str,
    user_prompt: str,
) -> list[tuple[str, str]]:
    return [
        ("system", system_prompt),
        ("user", user_prompt),
    ]


def image_message_from_path(prompt: str, image_path: str) -> list[dict[str, Any]]:
    path = Path(image_path)
    suffix = path.suffix.lower().lstrip(".") or "png"
    image_bytes = path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/{suffix};base64,{image_b64}"},
        },
        {"type": "text", "text": prompt},
    ]


def print_message(message: AIMessage | AIMessageChunk) -> None:
    print("content:")
    print(message.text)
    reasoning = message.additional_kwargs.get("reasoning_content")
    if reasoning:
        print("\nreasoning_content:")
        print(reasoning)
    if message.tool_calls:
        print("\ntool_calls:")
        for tool_call in message.tool_calls:
            print(tool_call)
    if message.usage_metadata:
        print("\nusage_metadata:")
        print(message.usage_metadata)


def stream_and_collect(
    model: ChatMoonshot,
    messages: list[BaseMessage | tuple[str, Any]],
) -> AIMessageChunk | None:
    full: AIMessageChunk | None = None
    printed_reasoning = False

    for chunk in model.stream(messages):
        if full is None:
            full = chunk
        else:
            full += chunk

        text = chunk.text
        if text:
            print(text, end="", flush=True)

        reasoning = chunk.additional_kwargs.get("reasoning_content")
        if reasoning:
            if not printed_reasoning:
                print("\n\n[reasoning stream]")
                printed_reasoning = True
            print(reasoning, end="", flush=True)

    print()
    return full


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default="kimi-k2.5",
        help="Moonshot model name.",
    )
    parser.add_argument(
        "--thinking",
        choices=["default", "on", "off"],
        default="default",
        help="Thinking mode for models that support it.",
    )
