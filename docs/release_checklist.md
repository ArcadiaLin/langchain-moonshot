# Release Checklist

This checklist is for publishing `langchain-moonshot` as a standalone LangChain integration package and preparing the corresponding docs PR.

## 1. Package readiness

Before publishing, make sure the package meets the baseline expected for a community integration:

- `ChatMoonshot` imports cleanly from `langchain_moonshot`
- `README.md` explains installation, credentials, basic usage, streaming, tools, structured output, and model-specific constraints
- unit tests, lint, and type checks all pass
- live integration tests pass against a real Moonshot API key
- version in `pyproject.toml` is updated for the intended release
- package metadata is accurate enough for PyPI consumers

Recommended local verification:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
env UV_CACHE_DIR=/tmp/uv-cache uv run mypy langchain_moonshot tests
env UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/unit_tests
RUN_MOONSHOT_INTEGRATION=1 env UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/integration_tests
```

## 2. Build the distribution

Build the wheel and source distribution locally:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv build
```

Expected output artifacts:

- `dist/langchain_moonshot-<version>.tar.gz`
- `dist/langchain_moonshot-<version>-py3-none-any.whl`

Inspect the built metadata before publishing:

```bash
tar -tzf dist/langchain_moonshot-*.tar.gz | sed -n '1,80p'
python -m zipfile -l dist/langchain_moonshot-*.whl | sed -n '1,80p'
```

## 3. Publish to PyPI

The supported path is to publish this package independently, not to submit it to the `langchain` monorepo.

### PyPI account setup

- Create a PyPI account
- Verify email
- Enable 2FA
- Create an API token scoped to the project or your account

### Publish with `uv`

Set credentials:

```bash
export UV_PUBLISH_USERNAME="__token__"
export UV_PUBLISH_PASSWORD="<your-pypi-token>"
```

Publish:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv publish
```

If you want to validate artifacts first, publish to TestPyPI before the real release.

## 4. Open a docs PR to `langchain-ai/docs`

After the package is live on PyPI, open a separate PR to the docs repository.

Recommended sequence:

1. Fork `langchain-ai/docs`
2. Create a branch such as `moonshot-chat-integration`
3. Copy the chat integration template from the docs repo
4. Adapt it using the draft in `docs/langchain_docs_moonshot.mdx`
5. Verify frontmatter, code blocks, links, and provider metadata
6. Open a docs-only PR

Important constraints from the LangChain guide:

- the package PR and the docs PR are separate
- docs examples must run
- frontmatter and Mintlify components must be valid
- localization requirements should be checked before opening the PR

## 5. Suggested release bundle

For a clean release, keep these together:

- one commit for runtime and test fixes
- one commit for README and release docs
- one commit for the docs-page draft if you want a clean cherry-pick into the docs repo

## 6. Optional but recommended

- add GitHub Actions for `ruff`, `mypy`, unit tests, and gated live integration tests
- add `project.urls` in `pyproject.toml` after repository and issue tracker URLs are finalized
- add a changelog or release notes entry once versioning becomes regular
