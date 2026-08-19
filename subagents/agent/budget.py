from __future__ import annotations

from dataclasses import dataclass

from ..errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class Budget:
    """What an agent is allowed to spend on one run: turns and tokens.

    Both bounds exist because they fail differently. Running out of steps
    means the model was still calling tools and never answered - the run is
    truncated but paid for, so it returns normally with a "step_budget" stop
    reason. Running out of tokens is a cost ceiling the caller set, so it
    raises TokenBudgetExceeded with the partial state attached.

    steps=1 makes the agent single-shot: one model call, one round of tools,
    and the model gets no turn to react to the results. Useful for a
    classify-or-extract step where reflection buys nothing, but note that a
    single-shot agent given tools will stop with "step_budget" rather than an
    answer whenever it calls one.

    Frozen so a Budget can be shared between agents without one run's limits
    being mutated out from under another.
    """

    steps: int = 10
    tokens: int | None = None

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ConfigurationError(
                f"Budget.steps must be at least 1, got {self.steps}"
            )
        if self.tokens is not None and self.tokens < 1:
            raise ConfigurationError(
                f"Budget.tokens must be at least 1 when set, got {self.tokens}"
            )
