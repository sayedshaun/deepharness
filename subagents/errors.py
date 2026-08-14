from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .providers.base import TokenUsage


class SubagentsError(Exception):
    """Base class for all errors raised by subagents."""


class ConfigurationError(SubagentsError):
    """Raised when an Agent or Toolbox is used in an invalid configuration."""


class ToolNotFoundError(SubagentsError, KeyError):
    """Raised when a Toolbox is asked for a tool that isn't registered."""


class ProviderError(SubagentsError):
    """Raised when an LLM provider request fails after retries."""


class ExecutionError(SubagentsError):
    """Raised when a node function raises during graph execution."""

    def __init__(self, node_name: str, original: Exception):
        self.node_name = node_name
        self.original = original
        super().__init__(f"Node '{node_name}' failed: {original!r}")


class TokenBudgetExceeded(SubagentsError):
    """Raised when an Agent's cumulative token usage exceeds its token_budget."""

    def __init__(self, agent_name: str, usage: TokenUsage, budget: int):
        self.agent_name = agent_name
        self.usage = usage
        self.budget = budget
        super().__init__(
            f"{agent_name} used {usage.total_tokens} tokens, exceeding its budget of {budget}"
        )
