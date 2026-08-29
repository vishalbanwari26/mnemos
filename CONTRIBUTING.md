# Contributing to mnemos

Thanks for considering a contribution. mnemos is a portfolio-scale project,
not a corporate one — issues and PRs are genuinely welcome, but response
time will vary.

## Before opening a PR

- **Bug fixes / small improvements**: just open a PR directly, no need to
  file an issue first.
- **New features or anything that changes the architecture** (a new
  `StorageBackend`, a new `LLMClient`, a new memory type): please open an
  issue first to discuss the approach. This project intentionally keeps a
  narrow, honestly-implemented scope (see "What's intentionally out of
  scope" in the README) — a design conversation up front saves both of us
  from a large PR that doesn't fit that scope.

## Development setup

See "Running it" in the [README](README.md) — `uv sync --extra dev`,
`alembic upgrade head`, `uv run pytest`. Tests are fast and fully offline
(mocked LLM/embeddings) except `tests/unit/test_retrieval.py`, which uses
real local embeddings on purpose — that file's whole point is proving
semantic ranking actually works.

## Before submitting

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

All three should be clean. CI runs the same checks, plus the dashboard's
`npm run lint` / `npm run build` if you touched `dashboard/`.

## Code style

- No comments explaining *what* code does — names should do that. Comments
  are for non-obvious *why* (a workaround, a constraint, a tradeoff).
- No em dashes in prose (README, docstrings, commit messages) — commas or
  colons instead.
- Don't add abstractions, config knobs, or error handling for scenarios
  that can't happen. Match the existing pattern in the module you're
  touching rather than introducing a new one.

## Reporting issues

Include: what you ran, what you expected, what happened, and your
`STORAGE_BACKEND`/`LLM_PROVIDER` if relevant (a lot of behavior is
backend-specific — see the storage comparison in the README for known
tradeoffs like Neo4j's eventual-consistency cost).
