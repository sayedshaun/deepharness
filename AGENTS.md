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

## Data Models

Use the right tool for the job:

- **`dataclass`** → internal/domain data, configuration, state, nodes, agent metadata.
- **Pydantic** → structured LLM output, external input/output, validation, serialization.

Prefer:

```python
@dataclass(slots=True)
class AgentConfig:
    name: str
```

## Architecture

Keep modules focused and responsibilities separate.

```text
agent/      → agent logic
graph/      → graph/node execution
...
```

Avoid large files, circular dependencies, and tightly coupled components.

## Agents

- Give each agent one clear responsibility.
- Keep orchestration separate from agent implementation.
- Prefer composition over complex inheritance.
- Keep public APIs small and explicit.

## HTTP

Use `httpx` for all HTTP communication.

Do not scatter raw HTTP calls throughout the application. Reuse clients where appropriate and handle timeouts/errors explicitly.

## Python

- Target **Python 3.10+**.
- Follow PEP 8.
- Add type hints to all new public functions and methods.
- Use `ruff format` and `ruff check`.
- Prefer standard-library solutions.

## Error Handling

- Catch specific exceptions.
- Never silently swallow errors.
- Do not expose secrets or sensitive data in logs.

## Testing

Test behavior and important edge cases. Keep tests deterministic and avoid unnecessary external dependencies.

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
- Do not include AI-generated signatures or co-author lines.
- Only commit when explicitly asked.

### Examples

- `feat: add configuration loader`
- `fix: correct login validation`
- `docs: update README`
- `refactor: rename database models`
- `fix: improve error handling`

## Before Finishing

- Code is formatted (`make fmt`)
- No unused imports or debug code
- Type hints added where appropriate
- Tests pass (`make test`)
- Commit message follows the Commit Rules above