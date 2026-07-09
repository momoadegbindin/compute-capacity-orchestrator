from __future__ import annotations

import pytest

from compute_capacity_orchestrator.schemas.resources import ClusterNode
from compute_capacity_orchestrator.schemas.schedule import SchedulingSnapshot, GpuAssignment, SchedulingDecision
from compute_capacity_orchestrator.schemas.workload import JobRequest


def _job(job_id: str, release_time: int = 0) -> JobRequest:
    return JobRequest(
        job_id=job_id,
        gpu_demand=2,
        duration=5,
        priority=10.0,
        release_time=release_time,
        deadline=20,
    )


def _node(node_id: str) -> ClusterNode:
    return ClusterNode(
        node_id=node_id,
        total_gpus=8,
        available_gpus=8,
        topology_group="rack-a",
    )


def test_scheduling_snapshot_validation() -> None:
    job = _job("job-001")
    node = _node("node-001")

    snapshot = SchedulingSnapshot(
        current_time=0,
        queued_jobs=(job,),
        nodes=(node,),
    )

    assert snapshot.current_time == 0
    assert snapshot.queued_jobs == (job,)
    assert snapshot.nodes == (node,)

    with pytest.raises(ValueError) as exc_info:
        SchedulingSnapshot(
            current_time=0,
            queued_jobs=(_job("job-001"), _job("job-001")),
            nodes=(node,),
        )

    assert "job_id" in str(exc_info.value)
    assert "unique" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        SchedulingSnapshot(
            current_time=0,
            queued_jobs=(job,),
            nodes=(_node("node-001"), _node("node-001")),
        )

    assert "node_id" in str(exc_info.value)
    assert "unique" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        SchedulingSnapshot(
            current_time=0,
            queued_jobs=(_job("job-002", release_time=3),),
            nodes=(node,),
        )

    assert "job-002" in str(exc_info.value)
    assert "release_time" in str(exc_info.value)



def test_gpu_assignment_validation() -> None:
    assignment = GpuAssignment(
        job_id="job-001",
        node_id="node-001",
        gpu_count=4,
    )

    assert assignment.job_id == "job-001"
    assert assignment.node_id == "node-001"
    assert assignment.gpu_count == 4

    with pytest.raises(ValueError) as exc_info:
        GpuAssignment(
            job_id="job-002",
            node_id="node-001",
            gpu_count=0,
        )

    assert "job-002" in str(exc_info.value)
    assert "node-001" in str(exc_info.value)
    assert "gpu_count" in str(exc_info.value)


def test_scheduling_decision_validation() -> None:
    assignment = GpuAssignment(
        job_id="job-001",
        node_id="node-001",
        gpu_count=4,
    )

    decision = SchedulingDecision(
        assignments=(assignment,),
        started_job_ids=("job-001",),
        waiting_job_ids=("job-002",),
        objective_value=25.0,
        solver_status="optimal",
        runtime_ms=12.5,
    )

    assert decision.assignments == (assignment,)
    assert decision.started_job_ids == ("job-001",)
    assert decision.waiting_job_ids == ("job-002",)
    assert decision.objective_value == 25.0
    assert decision.solver_status == "optimal"
    assert decision.runtime_ms == 12.5

    with pytest.raises(ValueError) as exc_info:
        SchedulingDecision(
            assignments=(assignment,),
            started_job_ids=("job-001",),
            waiting_job_ids=("job-001",),
            objective_value=25.0,
            solver_status="optimal",
            runtime_ms=12.5,
        )

    assert "both started and waiting" in str(exc_info.value)