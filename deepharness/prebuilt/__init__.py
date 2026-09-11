"""Workflows assembled from agent/ and graph/, ready to run.

Each one is a shape that keeps getting rebuilt on top of `Agent` - packaged
here so callers can take it whole, and read it as a worked example when they
need a different shape instead.
"""

from .research import (
    DeepResearch,
    Finding,
    Planned,
    Planning,
    Researched,
    ResearchEvent,
    ResearchFinished,
    Researching,
    ResearchResult,
    Synthesizing,
)

__all__ = [
    "DeepResearch",
    "Finding",
    "Planned",
    "Planning",
    "ResearchEvent",
    "ResearchFinished",
    "ResearchResult",
    "Researched",
    "Researching",
    "Synthesizing",
]
