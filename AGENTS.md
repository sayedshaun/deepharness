# AGENTS.md

## Principles

- Keep the code **minimal, modular, readable, and easy to understand**.
- Prefer simple solutions over unnecessary abstractions.
- Make the smallest change necessary.
- Avoid unrelated refactoring.

## Dependencies

Runtime dependencies are strictly limited to:

- `httpx` — HTTP requests
- `pydantic` — validation and structured output

**Do not add other dependencies.** Use the Python standard library whenever possible.

## Architecture

Keep modules focused and responsibilities separate.

```text
agent/      → agent think/act loop, Toolbox, @tool
graph/      → graph/node/edge definitions, executor (waves + merging)
providers/  → LLM interface, HTTPClient, wire types, per-vendor clients
tools/      → built-in tool implementations
```

Avoid large files, circular dependencies, and tightly coupled components.

## Agents

- Give each agent one clear responsibility.
- Keep orchestration separate from agent implementation.
- Prefer composition over complex inheritance.
- Keep public APIs small and explicit.

## Code Style

- Target **Python 3.11+** and set `requires-python = ">=3.11"` in `pyproject.toml`.
- Follow **PEP 8**.
- Format and lint with **Ruff** (`make fmt`) before committing. Do not use Black.
- Add **type hints** to all new public functions and methods.
- Write docstrings that explain **why** something exists or behaves a certain way, rather than restating what the code already makes clear.
- Keep inline comments to a minimum. Add comments only for **non-obvious constraints, workarounds, or design decisions**.
- Prefer modern Python features and standard-library tools where appropriate:
  - `pathlib` for filesystem operations.
  - `dataclasses` for structured data and simple value objects.
  - Modern type-hinting syntax.
- Use `@dataclass` when an object primarily represents **data rather than behavior**.
- Use `@dataclass(slots=True)` when there is a clear benefit, such as creating many instances, reducing memory usage, or preventing accidental attributes. Do not use `slots=True` by default.
- Use regular classes when an object primarily represents **behavior, lifecycle, or complex state management**.
- Keep implementations **simple, readable, and maintainable**. Prefer straightforward solutions over unnecessary abstractions or clever patterns.
- Avoid premature optimization and abstractions for hypothetical future requirements. Optimize for **clarity and maintainability today**.

Prefer:

```python
@dataclass(slots=True)
class AgentConfig:
    name: str
```

## HTTP

Use `httpx` for all HTTP communication.

Do not scatter raw HTTP calls throughout the application. Reuse clients where appropriate and handle timeouts/errors explicitly.

## Error Handling

- Catch specific exceptions.
- Never silently swallow errors.
- Do not expose secrets or sensitive data in logs.

## Testing

- Test behavior and important edge cases, not implementation details.
- Keep tests deterministic; no real network calls — inject an `httpx` client/transport instead.
- Run with `make test`.

## Avoid

- Unnecessary dependencies
- Over-engineering
- Giant classes/modules
- Deep inheritance
- Global mutable state
- Premature abstractions
- Pydantic for simple internal data

## Golden Rule

> **Simple, modular, typed, dependency-light code.**

## Commit Rules

- Write concise, one-line commit messages, prefixed with a Conventional Commits type: `feat:`, `fix:`, `refactor:`, `style:`, `docs:`, `test:`, `chore:`.
- After the prefix, use the imperative mood (e.g., "add", "fix", "update", "refactor").
- Keep each commit focused on a single logical change.
- When multiple files are staged for commit, commit one file at a time rather than bundling them into a single commit, unless the files together form one indivisible change (e.g., an implementation and its own test file).
- Do not include AI-generated signatures or co-author lines.
- Only commit when explicitly asked.

### Examples

- `feat: add configuration loader`
- `fix: correct login validation`
- `docs: update README`
- `refactor: rename database models`
- `fix: improve error handling`

## Before Finishing

- Code is formatted and linted (`make fmt`)
- No unused imports or debug code
- Type hints added where appropriate
- Tests pass (`make test`)
- Commit message follows the Commit Rules above
