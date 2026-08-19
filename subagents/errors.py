from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .providers.base import TokenUsage


class SubagentsError(Exception):
    """Base class for all errors raised by subagents."""


class ConfigurationError(SubagentsError):
    """Raised when an Agent or Toolbox is used in an invalid configuration."""


class ToolNotFoundError(SubagentsError, KeyError):
    """Raised when a Toolbox is asked for a tool that isn't registered."""


class OutputValidationError(SubagentsError):
    """Raised when a model's structured answer does not fit the output= shape.

    Handed back to the model as the failing call's result rather than ending
    the run, so it can correct the fields and answer again.
    """


class ProviderError(SubagentsError):
    """Raised when an LLM provider request fails after retries."""


class HumanInputRequired(SubagentsError):
    """Raised by a tool to pause the agent and wait for a human answer."""

    def __init__(self, question: str):
        self.question = question
        super().__init__(question)


class ExecutionError(SubagentsError):
    """Raised when a node function raises during graph execution."""

    def __init__(self, node_name: str, original: Exception):
        self.node_name = node_name
        self.original = original
        super().__init__(f"Node '{node_name}' failed: {original!r}")


class ConcurrentUpdateError(SubagentsError):
    """Raised when parallel branches write the same state field with no reducer.

    Silently picking a winner would drop one branch's work, so the graph
    refuses to guess: declare a reducer on the field, or give each branch its
    own field.
    """

    def __init__(self, field_name: str, writer_count: int):
        self.field_name = field_name
        self.writer_count = writer_count
        super().__init__(
            f"{writer_count} concurrent branches wrote '{field_name}' and it "
            f"declares no reducer; add field(metadata={{'reducer': ...}}) to it "
            f"or give each branch its own field"
        )


class StepLimitExceeded(SubagentsError):
    """Raised when a graph run exceeds max_steps, usually a loop that never exits.

    state carries the partial result so a run that hits the limit can still
    be inspected, matching TokenBudgetExceeded.
    """

    def __init__(self, limit: int, state: Any | None = None):
        self.limit = limit
        self.state = state
        super().__init__(f"Graph exceeded its limit of {limit} steps")


class TokenBudgetExceeded(SubagentsError):
    """Raised when an Agent's cumulative token usage exceeds its Budget.tokens.

    state carries the agent's partial result, so the tokens already paid for
    aren't lost with the exception - inspect it, or resume from its messages.
    """

    def __init__(
        self,
        agent_name: str,
        usage: TokenUsage,
        budget: int,
        state: dict[str, Any] | None = None,
    ):
        self.agent_name = agent_name
        self.usage = usage
        self.budget = budget
        self.state = state
        super().__init__(
            f"{agent_name} used {usage.total_tokens} tokens, exceeding its budget of {budget}"
        )
