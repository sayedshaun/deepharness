"""What a run may do on its own, what it must ask about, and what it may not.

`requires_approval` on a tool answers this for the tool as a whole, which is
too coarse once the tool is `run_command`: reading a file and deleting a
directory are the same tool. A policy decides per call, from the arguments the
model actually sent, and lives outside the tool so the same toolbox can be
trusted differently in two places.
"""

from __future__ import annotations

from collections.abc import Iterable
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

    def matches(self, name: str, arguments: dict[str, Any]) -> bool:
        if not fnmatchcase(name, self.tool):
            return False
        for key, pattern in (self.arguments or {}).items():
            if key not in arguments:
                return False
            if not fnmatchcase(str(arguments[key]), pattern):
                return False
        return True


class Permissions:
    """A run's rules, most restrictive first.

    `deny` wins over `allow`, which wins over `ask`; a call no rule matches gets
    no opinion, and falls back to the tool's own `requires_approval`. That
    ordering is what makes a policy safe to widen: adding an `allow` can never
    quietly override a `deny` already written down.

    Rules are given as tool-name patterns or `Rule`s:

        Permissions(
            allow=["read_file", "list_files", Rule("run_command", {"command": "git *"})],
            deny=[Rule("run_command", {"command": "*rm -rf*"})],
            ask=["write_file", "edit_file"],
        )
    """

    __slots__ = ("_allow", "_ask", "_deny")

    def __init__(
        self,
        *,
        allow: Iterable[str | Rule] = (),
        ask: Iterable[str | Rule] = (),
        deny: Iterable[str | Rule] = (),
    ) -> None:
        self._deny = _rules(deny)
        self._allow = _rules(allow)
        self._ask = _rules(ask)

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


def _rules(rules: Iterable[str | Rule]) -> tuple[Rule, ...]:
    return tuple(rule if isinstance(rule, Rule) else Rule(rule) for rule in rules)
