# Changelog

All notable changes to `langchain-moonshot` will be documented in this file.

## 0.1.0 - 2026-04-06

- Rebuilt the package around `ChatMoonshot` as a standalone LangChain integration.
- Marked this release as a breaking rewrite relative to the previous `0.0.x` PyPI releases.
- Added support for Moonshot-specific chat parameters such as `thinking`,
  `prompt_cache_key`, `safety_identifier`, and `max_completion_tokens`.
- Added standard LangChain coverage for tool calling, structured output,
  multimodal inputs, streaming, async, and VCR-backed integration checks.
- Raised the supported Python range to `>=3.10,<4.0`.
- Added packaging metadata, release documentation, and CI for PyPI publishing.
- Moved the project homepage from
  `https://github.com/RyanFeiluX/langchain_moonshot` to
  `https://github.com/ArcadiaLin/langchain-moonshot`.

## 0.0.7 - 2024-03-29

- Published on PyPI as a wheel-only release; no source distribution was listed.
- PyPI project metadata listed author `Ryan Xiao` and Python support
  `>=3.8.1,<4.0`.
- PyPI maintainers were `ArcadiaLin` and `RyanXiao007`.
- The PyPI README still described the older OpenAI-style wrappers
  `OpenAI`, `ChatOpenAI`, and `OpenAIEmbeddings`.
- Project links on PyPI pointed to the previous homepage
  `https://github.com/RyanFeiluX/langchain_moonshot`.

## 0.0.6 - 2024-03-29

- Earliest release currently visible on PyPI.
- Published on PyPI as a wheel-only release; no source distribution was listed.
- PyPI project metadata listed author `Ryan Xiao` and Python support
  `>=3.8.1,<4.0`.
- PyPI maintainer was `RyanXiao007`.
- The PyPI README described the older OpenAI-style wrappers `OpenAI`,
  `ChatOpenAI`, and `OpenAIEmbeddings`.
- Project links on PyPI pointed to the previous homepage
  `https://github.com/RyanFeiluX/langchain_moonshot`.
