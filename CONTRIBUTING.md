# Contributing

Thanks for your interest in DeepHarness. This file covers the local setup; the
conventions the code is held to live in [AGENTS.md](AGENTS.md).

## Setup

```bash
pip install -e ".[dev]"
```

## Everyday commands

```bash
make fmt     # ruff format, then ruff check
make test    # pytest -q
```

Run `make fmt` and `make test` before opening a pull request. CI runs the same
checks on Python 3.11, 3.12 and 3.13.

## Docs

Docs are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/):

```bash
pip install -e ".[docs]"
make docs     # serves at http://127.0.0.1:8000
```

## Pull requests

- Keep each commit focused on a single logical change.
- Write one-line commit messages prefixed with a Conventional Commits type:
  `feat:`, `fix:`, `refactor:`, `style:`, `docs:`, `test:`, `chore:`.
- Add tests for new behaviour and edge cases. Tests must be deterministic and
  must not make real network calls — inject an `httpx` client or transport.
- Runtime dependencies are limited to `httpx`; prefer the standard library.
