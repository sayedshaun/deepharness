"""What a run may do on its own, what it must ask about, and what it may not.

`requires_approval` on a tool answers this for the tool as a whole, which is
too coarse once the tool is `run_command`: reading a file and deleting a
directory are the same tool. A policy decides per call, from the arguments the
model actually sent, and lives outside the tool so the same toolbox can be
trusted differently in two places.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Literal

from ..errors import ConfigurationError

Decision = Literal["allow", "ask", "deny"]
"""What may happen to one call: run it, pause for a human, or refuse it."""


@dataclass(frozen=True, slots=True)
class Rule:
    """A tool-name pattern, optionally narrowed to certain arguments.

    Both sides are glob patterns (`fnmatch`), matched case-sensitively:
    `Rule("run_command", {"command": "git log*"})` covers reading history
    without covering `git push`. An argument the rule does not mention is
    unconstrained, and a rule mentioning an argument the call did not send does
    not match - an absent argument cannot be vouched for.
    """

    tool: str
    arguments: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.tool:
            raise ConfigurationError("Rule.tool must name a tool or a pattern")

    @property
    def is_pattern(self) -> bool:
        """Whether this rule matches by shape rather than naming one tool.

        A rule naming a tool can be checked against a toolbox; a pattern can
        only be checked against a call, so the two are validated differently.
        """
        return any(char in self.tool for char in "*?[")

    def matches(self, name: str, arguments: dict[str, Any]) -> bool:
        if not fnmatchcase(name, self.tool):
            return False
        for key, pattern in (self.arguments or {}).items():
            if key not in arguments:
                return False
            if not fnmatchcase(str(arguments[key]), pattern):
                return False
        return True


RuleLike = str | Rule | Callable[..., Any]
"""How a rule may be written: the tool itself, a name pattern, or a Rule."""


class Permissions:
    """A run's rules, most restrictive first.

    `deny` wins over `allow`, which wins over `ask`; a call no rule matches gets
    no opinion, and falls back to the tool's own `requires_approval`. That
    ordering is what makes a policy safe to widen: adding an `allow` can never
    quietly override a `deny` already written down.

    A rule is given as a tool, a name pattern, or a `Rule`:

        Permissions(
            allow=[read_file, list_files, Rule("run_command", {"command": "git *"})],
            deny=[Rule("run_command", {"command": "*rm -rf*"})],
            ask=[FileTool.WRITE, FileTool.EDIT],
        )

    Passing the tool itself is worth preferring where it is in scope: an editor
    renames it with the function, and a typo is a NameError rather than a rule
    that silently matches nothing.
    """

    __slots__ = ("_allow", "_ask", "_deny")

    def __init__(
        self,
        *,
        allow: Iterable[RuleLike] = (),
        ask: Iterable[RuleLike] = (),
        deny: Iterable[RuleLike] = (),
    ) -> None:
        self._deny = _rules(deny)
        self._allow = _rules(allow)
        self._ask = _rules(ask)

    @property
    def rules(self) -> tuple[Rule, ...]:
        """Every rule, whatever its decision."""
        return (*self._deny, *self._allow, *self._ask)

    @property
    def gates(self) -> tuple[Rule, ...]:
        """The rules whose silence would be unsafe: everything that refuses or asks.

        An allow rule that matches nothing is inert - the call falls back to the
        tool's own requires_approval - so a policy shared between agents may
        name tools some of them do not have. A deny or ask rule that matches
        nothing is the opposite: the call it was written to stop runs.
        """
        return (*self._deny, *self._ask)

    def decide(self, name: str, arguments: dict[str, Any]) -> Decision | None:
        """What to do with one call, or None when no rule has an opinion."""
        for decision, rules in (
            ("deny", self._deny),
            ("allow", self._allow),
            ("ask", self._ask),
        ):
            if any(rule.matches(name, arguments) for rule in rules):
                return decision  # type: ignore[return-value]
        return None


def _rules(rules: Iterable[RuleLike]) -> tuple[Rule, ...]:
    return tuple(_rule(rule) for rule in rules)


def _rule(rule: RuleLike) -> Rule:
    """One rule from whichever form the caller found convenient.

    A callable is read for the name it is registered under rather than its
    __name__, so a tool renamed with @tool(name=...) is still matched by the
    name the model actually sees.
    """
    if isinstance(rule, Rule):
        return rule
    if isinstance(rule, str):
        return Rule(rule)
    if callable(rule):
        spec = getattr(rule, "_tool_spec", None)
        return Rule(spec.name if spec is not None else rule.__name__)
    raise ConfigurationError(
        f"a permission rule must be a tool, a name pattern or a Rule, "
        f"got {type(rule).__name__}"
    )
