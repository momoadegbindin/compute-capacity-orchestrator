"""Base scheduler interface."""

from __future__ import annotations

from typing import Protocol

from compute_capacity_orchestrator.schemas.schedule import (
    SchedulingDecision,
    SchedulingSnapshot,
)


class Scheduler(Protocol):
    """Interface implemented by all scheduling engines."""

    def solve(self, snapshot: SchedulingSnapshot) -> SchedulingDecision:
        """Return a scheduling decision for the given snapshot."""
        ...