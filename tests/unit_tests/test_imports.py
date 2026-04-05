from langchain_moonshot import __all__

EXPECTED_ALL = [
    "ChatMoonshot",
    "__version__",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
