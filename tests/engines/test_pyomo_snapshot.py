from __future__ import annotations

import pytest
import pyomo.environ as pyo

from compute_capacity_orchestrator.engines.pyomo_snapshot import PyomoSnapshotScheduler
from compute_capacity_orchestrator.engines.validation import validate_decision
from compute_capacity_orchestrator.experiments.scenarios import (
    build_greedy_vs_mip_snapshot,
    build_deadline_pressure_snapshot,
)

def _skip_if_highs_unavailable() -> None:
    solver = pyo.SolverFactory("appsi_highs")
    if not solver.available(exception_flag=False):
        pytest.skip("appsi_highs solver is not available")


def test_pyomo_snapshot_scheduler_finds_global_snapshot_solution() -> None:
    _skip_if_highs_unavailable()

    # Greedy-vs-MIP snapshot:
    # 2 nodes, 4 queued jobs, 8 available GPUs.
    # Exact MIP should start train-small and train-large for objective 48.0.
    snapshot = build_greedy_vs_mip_snapshot()

    scheduler = PyomoSnapshotScheduler(deadline_penalty_weight=0.0)
    decision = scheduler.solve(snapshot)

    validate_decision(snapshot, decision)

    assert set(decision.started_job_ids) == {"train-small", "train-large"}
    assert set(decision.waiting_job_ids) == {
        "eval-batch",
        "low-priority-sweep",
    }
    assert decision.objective_value == pytest.approx(48.0)
    assert decision.solver_status == "optimal"
    assert decision.runtime_ms >= 0.0


def test_pyomo_snapshot_scheduler_accounts_for_waiting_deadline_penalty() -> None:
    _skip_if_highs_unavailable()

    # Deadline-pressure snapshot:
    # 1 available GPU, 2 jobs.
    # The exact model should start deadline-risk because leaving it waiting is costlier.
    snapshot = build_deadline_pressure_snapshot()

    scheduler = PyomoSnapshotScheduler(
        deadline_penalty_weight=10.0,
        decision_step=1,
    )

    decision = scheduler.solve(snapshot)

    validate_decision(snapshot, decision)

    assert decision.started_job_ids == ("deadline-risk",)
    assert decision.waiting_job_ids == ("high-priority-normal",)

    # If deadline-risk starts:
    #   started priority = 6
    #   started deadline penalty = 10 * max(0, 0 + 5 - 3) = 20
    #   waiting deadline penalty for high-priority-normal = 0
    #   objective = 6 - 20 - 0 = -14
    #
    # If high-priority-normal starts:
    #   started priority = 15
    #   started deadline penalty = 0
    #   waiting deadline penalty for deadline-risk =
    #       10 * max(0, 0 + 1 + 5 - 3) = 30
    #   objective = 15 - 0 - 30 = -15
    #
    # The model chooses -14 over -15.
    assert decision.objective_value == pytest.approx(-14.0)