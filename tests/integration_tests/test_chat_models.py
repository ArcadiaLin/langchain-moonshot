from __future__ import annotations

import os

import pytest

from langchain_moonshot import ChatMoonshot

pytestmark = pytest.mark.skipif(
    not (
        os.getenv("RUN_MOONSHOT_INTEGRATION") == "1"
        and os.getenv("MOONSHOT_API_KEY")
    ),
    reason=(
        "Set RUN_MOONSHOT_INTEGRATION=1 and MOONSHOT_API_KEY to run live "
        "Moonshot integration tests."
    ),
)


def test_chat_moonshot_invoke_smoke() -> None:
    model = ChatMoonshot(
        model="kimi-k2.5",
        api_key=os.environ["MOONSHOT_API_KEY"],
        base_url=os.getenv("MOONSHOT_API_BASE", "https://api.moonshot.cn/v1"),
        temperature=1.0,
    )

    message = model.invoke("Reply with the single word pong.")

    assert isinstance(message.content, str)
    assert message.content
