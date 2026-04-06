# Changelog

All notable changes to `langchain-moonshot` will be documented in this file.

## 0.1.0 - 2026-04-06

- Rebuilt the package around `ChatMoonshot` as a standalone LangChain integration.
- Added support for Moonshot-specific chat parameters such as `thinking`,
  `prompt_cache_key`, `safety_identifier`, and `max_completion_tokens`.
- Added standard LangChain coverage for tool calling, structured output,
  multimodal inputs, streaming, async, and VCR-backed integration checks.
- Raised the supported Python range to `>=3.10,<4.0`.
- Added packaging metadata, release documentation, and CI for PyPI publishing.
