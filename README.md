# LangChain Moonshot

`langchain-moonshot` is a standalone LangChain integration package for Moonshot AI chat models.

It provides a `ChatMoonshot` implementation built on top of `langchain-openai`, with Moonshot-specific support for:

- `reasoning_content` in non-streaming and streaming responses
- `thinking`, `prompt_cache_key`, `safety_identifier`, and `max_completion_tokens`
- Moonshot model capability profiles
- tool calling, structured output, and multimodal inputs supported by Moonshot models

## Overview

### Integration details

| Class | Package | Status | Notes |
| :--- | :--- | :---: | :--- |
| `ChatMoonshot` | `langchain-moonshot` | alpha | Standalone provider package for Moonshot AI |

### Model features

| Tool calling | Structured output | Image input | Video input | Token-level streaming | Native async | Token usage | Reasoning traces |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## Installation

```bash
pip install -U langchain-moonshot
```

For local development:

```bash
uv sync --group dev --group test_integration
```

## Credentials

Create a Moonshot API key in the Moonshot console, then set `MOONSHOT_API_KEY`.

```python
import getpass
import os

if not os.getenv("MOONSHOT_API_KEY"):
    os.environ["MOONSHOT_API_KEY"] = getpass.getpass("Enter your Moonshot API key: ")
```

If you are using Moonshot's China endpoint explicitly, you can also set:

```python
os.environ.setdefault("MOONSHOT_API_BASE", "https://api.moonshot.cn/v1")
```

To enable LangSmith tracing:

```python
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
```

## Instantiation

```python
from langchain_moonshot import ChatMoonshot

llm = ChatMoonshot(
    model="kimi-k2.5",
    thinking=True,
    temperature=1.0,
    max_retries=2,
)
```

## Invocation

```python
messages = [
    ("system", "You are a concise bilingual assistant."),
    ("human", "Summarize why Moonshot reasoning models are useful in two bullet points."),
]

ai_msg = llm.invoke(messages)

print(ai_msg.text)
print(ai_msg.additional_kwargs.get("reasoning_content"))
print(ai_msg.usage_metadata)
```

## Streaming

```python
streaming_llm = ChatMoonshot(
    model="kimi-k2.5",
    thinking=True,
    temperature=1.0,
    stream_usage=True,
)

full = None
for chunk in streaming_llm.stream("Explain streaming output in two short bullet points."):
    if full is None:
        full = chunk
    else:
        full += chunk

    if chunk.text:
        print(chunk.text, end="")

    reasoning = chunk.additional_kwargs.get("reasoning_content")
    if reasoning:
        print(f"\n[reasoning] {reasoning}", end="")

print()
print(full.usage_metadata if full is not None else None)
```

## Tool Calling

```python
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


llm_with_tools = ChatMoonshot(
    model="kimi-k2.5",
    thinking=False,
    temperature=0.6,
).bind_tools([add, multiply])

messages = [
    ("system", "Use the provided math tools before answering."),
    ("human", "Add 17 and 25, multiply 12 by 13, then summarize the results."),
]

response = llm_with_tools.invoke(messages)
print(response.tool_calls)

if response.tool_calls:
    tool_results = []
    for tool_call in response.tool_calls:
        if tool_call["name"] == "add":
            result = add.invoke(tool_call["args"])
        else:
            result = multiply.invoke(tool_call["args"])
        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    final_response = llm_with_tools.invoke([*messages, response, *tool_results])
    print(final_response.text)
```

## Structured Output

Moonshot does not expose a distinct `json_schema` steering path in this package. `ChatMoonshot.with_structured_output(..., method="json_schema")` is intentionally downgraded to `function_calling`.

```python
from pydantic import BaseModel, Field


class WeatherAnswer(BaseModel):
    city: str = Field(description="City name")
    summary: str = Field(description="One-sentence weather summary")


structured_llm = ChatMoonshot(
    model="kimi-k2.5",
    thinking=False,
    temperature=0.6,
).with_structured_output(WeatherAnswer)

result = structured_llm.invoke("Summarize today's weather in Shanghai.")
print(result)
```

## Multimodal Input

Vision-capable Moonshot models accept OpenAI-style `image_url` content blocks.

```python
from langchain_core.messages import HumanMessage

vision_llm = ChatMoonshot(
    model="moonshot-v1-32k-vision-preview",
)

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe the image and mention one concrete detail."},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,<your-base64-image>"},
        },
    ]
)

response = vision_llm.invoke([message])
print(response.text)
```

## LangGraph Compatibility

`ChatMoonshot` works as a normal LangChain chat model inside LangGraph agent loops.

```python
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


tools = [add, multiply]
model = ChatMoonshot(
    model="kimi-k2.5",
    thinking=False,
    temperature=0.6,
).bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    response = model.invoke([("system", "Use tools when needed."), *state["messages"]])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "continue" if getattr(last, "tool_calls", None) else "end"


graph = StateGraph(AgentState)
graph.add_node("agent", model_call)
graph.add_node("tools", ToolNode(tools=tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "agent")

result = graph.compile().invoke(
    {"messages": [("user", "Add 3 and 4, multiply 12 by 13, then summarize.")]}
)
print(result["messages"][-1].content)
```

## Moonshot-Specific Notes

- `kimi-k2.5` is validated more strictly than generic OpenAI-compatible models.
- When `thinking=True`, `kimi-k2.5` expects `temperature=1.0`.
- When `thinking=False`, `kimi-k2.5` expects `temperature=0.6`.
- For `kimi-k2.5`, `top_p` must remain `0.95`, `n` must remain `1`, and both penalties must remain `0.0`.
- Forced tool usage via `tool_choice="required"` is rejected by this integration.
- Moonshot builtin `$web_search` is rejected when `thinking=True`.

## Local Verification

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
env UV_CACHE_DIR=/tmp/uv-cache uv run mypy langchain_moonshot tests
env UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/unit_tests
RUN_MOONSHOT_INTEGRATION=1 env UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/integration_tests
```

Manual smoke scripts remain available under [`scripts/`](./scripts) for exploratory checks across streaming, tools, vision, model matrices, and LangGraph agent flows.
