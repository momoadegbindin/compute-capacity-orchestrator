from __future__ import annotations

import pytest

from compute_capacity_orchestrator.engines.greedy import GreedyScheduler
from compute_capacity_orchestrator.engines.pyomo_snapshot import PyomoSnapshotScheduler

from app.views.snapshot_view import build_scheduler


def test_build_scheduler_returns_greedy_scheduler() -> None:
    scheduler = build_scheduler(
        scheduler_name="Greedy value density",
        deadline_penalty_weight=1.0,
    )

    assert isinstance(scheduler, GreedyScheduler)


def test_build_scheduler_returns_pyomo_scheduler_with_limits() -> None:
    scheduler = build_scheduler(
        scheduler_name="Exact MIP snapshot",
        deadline_penalty_weight=2.0,
        time_limit_seconds=15,
        relative_gap=0.01,
    )

    assert isinstance(scheduler, PyomoSnapshotScheduler)
    assert scheduler.deadline_penalty_weight == 2.0
    assert scheduler.decision_step == 1
    assert scheduler.time_limit_seconds == 15
    assert scheduler.relative_gap == 0.01
    assert scheduler.log_diagnostics is False


def test_build_scheduler_rejects_unknown_scheduler() -> None:
    with pytest.raises(ValueError, match="Unknown scheduler"):
        build_scheduler(
            scheduler_name="unknown",
            deadline_penalty_weight=0.0,
        )