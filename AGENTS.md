# AGENTS.md

## Principles

- Keep the code **minimal, modular, readable, and easy to understand**.
- Prefer simple solutions over unnecessary abstractions.
- Make the smallest change necessary.
- Avoid unrelated refactoring.

## Dependencies

Runtime dependencies are strictly limited to:

- `httpx` — HTTP requests

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

## Object-Oriented Design

Object orientation is a tool for keeping boundaries clear, not a goal in itself.
Use it where it earns its place, and follow these rules when you do.

### Encapsulation

- Keep state private (`_name`) and expose behaviour, not internals.
- An object should enforce its own invariants in `__init__` (or `__post_init__`);
  never rely on callers to keep it consistent.
- Do not put IO — printing, logging, filesystem, network — inside a domain
  object. Return a value or accept an injected callable instead.

### Abstraction and dependencies

- Depend on an interface, never on a concrete implementation. Core code
  (`agent/`, `graph/`) must not import a vendor module.
- Define interfaces with `abc.ABC` and `@abstractmethod`, and keep them narrow —
  an interface is the smallest set of methods callers actually need.
- One abstraction per seam. Do not add an interface until there is a second
  implementation or a test that needs to substitute one.

### Inheritance

- **Prefer composition.** Inherit only to declare a subtype relationship, never
  to reuse code — extract a function or collaborator for that.
- A subclass must be substitutable for its base: same contract, no stricter
  preconditions, no removed behaviour. A subclass that only overrides class-level
  configuration (endpoint, credential key) is fine; one that changes what a
  method means is not.
- One level deep. If you need two, the design is wrong.
- Do not subclass builtins (`dict`, `list`, `str`) for convenience — it inherits
  an entire mutable API you cannot constrain. Wrap or use a dataclass, and add a
  `to_dict()` where a plain mapping is required at a boundary.

### Polymorphism over duplication

- When two classes implement the same steps in the same order and differ only in
  details, put the sequence in the base class and let subclasses override the
  varying steps (template method). Do not copy the sequence per subclass.
- Prefer overriding a method to branching on a type or a string tag.

### Objects over loose dicts

- Model data that crosses a public boundary as a typed object (`@dataclass`),
  not a `dict[str, Any]` with string keys.
- Keep `dict[str, Any]` for wire payloads and JSON schemas, where the untyped
  shape is the format itself.

## Agents

- Give each agent one clear responsibility.
- Keep orchestration separate from agent implementation.
- Prefer composition over complex inheritance.
- Keep public APIs small and explicit.
- Inject collaborators (model, toolbox, budget) through `__init__`; do not
  construct them internally or read them from global state.

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
- Inheritance used for code reuse
- Subclassing builtins (`dict`, `list`, `str`)
- Duplicated method bodies that differ only in a constant
- Stringly-typed dicts on public APIs
- Global mutable state
- Premature abstractions
- Third-party validation libraries; dataclasses plus explicit parsing instead

## Golden Rule

> **Simple, modular, typed, dependency-light code — composed objects with
> clear boundaries, not deep hierarchies.**

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
