from __future__ import annotations

import pytest

from compute_capacity_orchestrator.engines.validation import validate_decision
from compute_capacity_orchestrator.schemas.resources import ClusterNode
from compute_capacity_orchestrator.schemas.schedule import (
    GpuAssignment,
    SchedulingDecision,
    SchedulingSnapshot,
)
from compute_capacity_orchestrator.schemas.workload import JobRequest


def test_validate_decision_accepts_feasible_decision() -> None:
    job_1 = JobRequest(
        job_id="job-001",
        gpu_demand=4,
        duration=10,
        priority=25.0,
        release_time=0,
        deadline=20,
    )
    job_2 = JobRequest(
        job_id="job-002",
        gpu_demand=2,
        duration=5,
        priority=10.0,
        release_time=0,
        deadline=15,
    )
    node = ClusterNode(
        node_id="node-001",
        total_gpus=8,
        available_gpus=8,
        topology_group="rack-a",
    )

    snapshot = SchedulingSnapshot(
        current_time=0,
        queued_jobs=(job_1, job_2),
        nodes=(node,),
    )

    decision = SchedulingDecision(
        assignments=(
            GpuAssignment(
                job_id="job-001",
                node_id="node-001",
                gpu_count=4,
            ),
        ),
        started_job_ids=("job-001",),
        waiting_job_ids=("job-002",),
        objective_value=25.0,
        solver_status="optimal",
        runtime_ms=4.2,
    )

    assert validate_decision(snapshot, decision) is None


def test_validate_decision_rejects_gpu_demand_mismatch() -> None:
    job = JobRequest(
        job_id="job-001",
        gpu_demand=4,
        duration=10,
        priority=25.0,
        release_time=0,
        deadline=20,
    )
    node = ClusterNode(
        node_id="node-001",
        total_gpus=8,
        available_gpus=8,
    )

    snapshot = SchedulingSnapshot(
        current_time=0,
        queued_jobs=(job,),
        nodes=(node,),
    )

    decision = SchedulingDecision(
        assignments=(
            GpuAssignment(
                job_id="job-001",
                node_id="node-001",
                gpu_count=3,
            ),
        ),
        started_job_ids=("job-001",),
        waiting_job_ids=(),
        objective_value=25.0,
        solver_status="optimal",
        runtime_ms=4.2,
    )

    with pytest.raises(ValueError) as exc_info:
        validate_decision(snapshot, decision)

    assert "job-001" in str(exc_info.value)
    assert "gpu_demand" in str(exc_info.value)


def test_validate_decision_rejects_node_capacity_violation() -> None:
    job_1 = JobRequest(
        job_id="job-001",
        gpu_demand=4,
        duration=10,
        priority=25.0,
        release_time=0,
        deadline=20,
    )
    job_2 = JobRequest(
        job_id="job-002",
        gpu_demand=5,
        duration=10,
        priority=20.0,
        release_time=0,
        deadline=20,
    )
    node = ClusterNode(
        node_id="node-001",
        total_gpus=8,
        available_gpus=8,
    )

    snapshot = SchedulingSnapshot(
        current_time=0,
        queued_jobs=(job_1, job_2),
        nodes=(node,),
    )

    decision = SchedulingDecision(
        assignments=(
            GpuAssignment(
                job_id="job-001",
                node_id="node-001",
                gpu_count=4,
            ),
            GpuAssignment(
                job_id="job-002",
                node_id="node-001",
                gpu_count=5,
            ),
        ),
        started_job_ids=("job-001", "job-002"),
        waiting_job_ids=(),
        objective_value=45.0,
        solver_status="optimal",
        runtime_ms=4.2,
    )

    with pytest.raises(ValueError) as exc_info:
        validate_decision(snapshot, decision)

    assert "node-001" in str(exc_info.value)
    assert "available_gpus" in str(exc_info.value)