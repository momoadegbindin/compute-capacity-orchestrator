from __future__ import annotations

import pytest

from compute_capacity_orchestrator.metrics.decision_metrics import (
    compute_decision_metrics,
)
from compute_capacity_orchestrator.schemas.resources import ClusterNode
from compute_capacity_orchestrator.schemas.schedule import (
    GpuAssignment,
    SchedulingDecision,
    SchedulingSnapshot,
)
from compute_capacity_orchestrator.experiments.scenarios import (
    build_started_and_waiting_lateness_snapshot,
)



def test_compute_decision_metrics_counts_started_and_waiting_lateness() -> None:
    # Started/waiting lateness snapshot:
    # one started late job, one waiting risky job; penalties are hand-computable.
    snapshot = build_started_and_waiting_lateness_snapshot()

    decision = SchedulingDecision(
        assignments=(
            GpuAssignment(
                job_id="started-late",
                node_id="node-001",
                gpu_count=2,
            ),
        ),
        started_job_ids=("started-late",),
        waiting_job_ids=("waiting-risky",),
        objective_value=20.0,
        solver_status="greedy_value_density",
        runtime_ms=3.5,
    )

    metrics = compute_decision_metrics(
        snapshot=snapshot,
        decision=decision,
        deadline_penalty_weight=2.0,
        decision_step=1,
    )

    assert metrics.jobs_submitted == 2
    assert metrics.jobs_started == 1
    assert metrics.jobs_waiting == 1
    assert metrics.gpu_capacity_available == 4
    assert metrics.gpu_capacity_used == 2
    assert metrics.gpu_utilization == pytest.approx(0.5)

    assert metrics.total_priority_started == 20.0

    # started-late: current_time + duration - deadline = 0 + 5 - 3 = 2
    assert metrics.started_deadline_penalty == 4.0

    # waiting-risky: current_time + decision_step + duration - deadline = 0 + 1 + 3 - 2 = 2
    assert metrics.waiting_deadline_penalty == 4.0

    assert metrics.deadline_penalty_incurred == 8.0
    assert metrics.objective_value == 12.0
    assert metrics.scheduler_objective_value == 20.0
    assert metrics.solver_status == "greedy_value_density"
    assert metrics.runtime_ms == 3.5


def test_compute_decision_metrics_rejects_invalid_penalty_settings() -> None:
    snapshot = SchedulingSnapshot(
        current_time=0,
        queued_jobs=(),
        nodes=(
            ClusterNode(
                node_id="node-001",
                total_gpus=4,
                available_gpus=4,
            ),
        ),
    )

    decision = SchedulingDecision(
        assignments=(),
        started_job_ids=(),
        waiting_job_ids=(),
        objective_value=0.0,
        solver_status="greedy_value_density",
        runtime_ms=0.0,
    )

    with pytest.raises(ValueError, match="deadline_penalty_weight"):
        compute_decision_metrics(
            snapshot=snapshot,
            decision=decision,
            deadline_penalty_weight=-1.0,
        )

    with pytest.raises(ValueError, match="decision_step"):
        compute_decision_metrics(
            snapshot=snapshot,
            decision=decision,
            decision_step=0,
        )