from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_openai.chat_models.base import BaseChatOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from langchain_moonshot import ChatMoonshot


def _create_model(**kwargs: Any) -> ChatMoonshot:
    return ChatMoonshot(
        model="kimi-k2.5",
        api_key="test-key",
        base_url="https://example.com/v1",
        **kwargs,
    )


def test_get_request_payload_adds_moonshot_extensions() -> None:
    model = _create_model(
        thinking=False,
        prompt_cache_key="session-1",
        safety_identifier="user-1",
        max_completion_tokens=256,
    )
    payload = model._get_request_payload(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ]
            ),
            AIMessage(
                content="prefill",
                additional_kwargs={
                    "partial": True,
                    "reasoning_content": "existing-thoughts",
                },
                name="assistant-name",
            ),
        ]
    )

    assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png"
    )
    assert payload["messages"][1]["partial"] is True
    assert payload["messages"][1]["reasoning_content"] == "existing-thoughts"
    assert payload["messages"][1]["name"] == "assistant-name"
    assert payload["extra_body"] == {
        "thinking": {"type": "disabled"},
        "prompt_cache_key": "session-1",
        "safety_identifier": "user-1",
    }
    assert payload["max_completion_tokens"] == 256
    assert "max_tokens" not in payload


def test_get_request_payload_rejects_invalid_k2_5_temperature() -> None:
    model = _create_model(temperature=0.2)

    with pytest.raises(ValueError, match="temperature=1.0"):
        model._get_request_payload("hello")


def test_get_request_payload_rejects_web_search_with_thinking() -> None:
    model = _create_model(thinking=True)

    with pytest.raises(ValueError, match="builtin \\$web_search"):
        model._get_request_payload(
            "hello",
            tools=[
                {
                    "type": "builtin_function",
                    "function": {"name": "$web_search"},
                }
            ],
        )


def test_bind_tools_rejects_required_tool_choice() -> None:
    model = _create_model()

    with pytest.raises(ValueError, match="tool_choice='required'"):
        model.bind_tools([], tool_choice="required")


def test_create_chat_result_preserves_reasoning_and_cached_tokens() -> None:
    model = _create_model()
    response = ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "kimi-k2.5",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                        "reasoning_content": "trace",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "cached_tokens": 5,
            },
        }
    )

    result = model._create_chat_result(response)
    message = result.generations[0].message

    assert message.content == "hello"
    assert message.additional_kwargs["reasoning_content"] == "trace"
    assert message.response_metadata["model_provider"] == "moonshot"
    assert message.usage_metadata["input_token_details"]["cache_read"] == 5
    assert result.llm_output["model_provider"] == "moonshot"
    assert (
        result.llm_output["token_usage"]["prompt_tokens_details"]["cached_tokens"] == 5
    )


def test_convert_chunk_to_generation_chunk_supports_reasoning_and_choice_usage(
) -> None:
    model = _create_model()
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "kimi-k2.5",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "step-1",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "add",
                                    "arguments": '{"a":',
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 4,
                        "total_tokens": 7,
                        "cached_tokens": 2,
                    },
                }
            ],
        }
    ).model_dump()

    generation_chunk = model._convert_chunk_to_generation_chunk(
        chunk,
        AIMessageChunk,
        {},
    )

    assert generation_chunk is not None
    assert generation_chunk.message.additional_kwargs["reasoning_content"] == "step-1"
    assert generation_chunk.message.response_metadata["model_provider"] == "moonshot"
    assert generation_chunk.message.tool_call_chunks[0]["name"] == "add"
    assert (
        generation_chunk.message.usage_metadata["input_token_details"]["cache_read"]
        == 2
    )


def test_with_structured_output_downgrades_json_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_with_structured_output(
        self: BaseChatOpenAI,
        schema: dict[str, Any] | type | None = None,
        *,
        method: str = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> str:
        captured["schema"] = schema
        captured["method"] = method
        captured["include_raw"] = include_raw
        captured["strict"] = strict
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(
        BaseChatOpenAI,
        "with_structured_output",
        fake_with_structured_output,
    )

    model = _create_model()
    result = model.with_structured_output(
        {"type": "object", "properties": {"answer": {"type": "string"}}},
        method="json_schema",
        include_raw=True,
        strict=True,
    )

    assert result == "sentinel"
    assert captured["method"] == "function_calling"
    assert captured["include_raw"] is True
    assert captured["strict"] is True
