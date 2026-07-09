from __future__ import annotations

import pytest

from compute_capacity_orchestrator.schemas.resources import ClusterNode
from compute_capacity_orchestrator.schemas.schedule import (
    GpuAssignment,
    SchedulingDecision,
)
from compute_capacity_orchestrator.schemas.workload import JobRequest
from compute_capacity_orchestrator.simulation.state import RunningJob, SimulationState
from compute_capacity_orchestrator.simulation.transition import (
    add_arrivals,
    advance_time,
    apply_scheduling_decision,
    build_scheduling_snapshot,
)


def _job(
    job_id: str,
    gpu_demand: int,
    duration: int = 5,
    priority: float = 10.0,
    release_time: int = 0,
    deadline: int = 20,
) -> JobRequest:
    return JobRequest(
        job_id=job_id,
        gpu_demand=gpu_demand,
        duration=duration,
        priority=priority,
        release_time=release_time,
        deadline=deadline,
    )


def _node(
    node_id: str = "node-001",
    total_gpus: int = 8,
    available_gpus: int = 8,
) -> ClusterNode:
    return ClusterNode(
        node_id=node_id,
        total_gpus=total_gpus,
        available_gpus=available_gpus,
    )


def test_build_scheduling_snapshot_uses_waiting_jobs_and_available_nodes() -> None:
    waiting_job = _job("waiting-job", gpu_demand=2)

    state = SimulationState(
        current_time=3,
        waiting_jobs=(waiting_job,),
        running_jobs=(),
        nodes=(_node(available_gpus=6),),
    )

    snapshot = build_scheduling_snapshot(state)

    assert snapshot.current_time == 3
    assert snapshot.queued_jobs == (waiting_job,)
    assert snapshot.nodes == state.nodes


def test_apply_scheduling_decision_starts_jobs_and_updates_capacity() -> None:
    job_a = _job("job-a", gpu_demand=2, duration=5)
    job_b = _job("job-b", gpu_demand=1, duration=3)

    state = SimulationState(
        current_time=0,
        waiting_jobs=(job_a, job_b),
        running_jobs=(),
        nodes=(_node(available_gpus=8),),
    )

    decision = SchedulingDecision(
        assignments=(
            GpuAssignment(
                job_id="job-a",
                node_id="node-001",
                gpu_count=2,
            ),
        ),
        started_job_ids=("job-a",),
        waiting_job_ids=("job-b",),
        objective_value=10.0,
        solver_status="manual",
        runtime_ms=0.0,
    )

    next_state = apply_scheduling_decision(state, decision)

    assert [job.job_id for job in next_state.waiting_jobs] == ["job-b"]
    assert [running.job.job_id for running in next_state.running_jobs] == ["job-a"]
    assert next_state.running_jobs[0].start_time == 0
    assert next_state.running_jobs[0].completion_time == 5
    assert next_state.nodes[0].available_gpus == 6


def test_apply_scheduling_decision_preserves_existing_running_jobs() -> None:
    existing_job = _job("already-running", gpu_demand=2, duration=10)
    new_job = _job("new-job", gpu_demand=1, duration=3)

    existing_running_job = RunningJob(
        job=existing_job,
        assignments=(
            GpuAssignment(
                job_id="already-running",
                node_id="node-001",
                gpu_count=2,
            ),
        ),
        start_time=0,
        completion_time=10,
    )

    state = SimulationState(
        current_time=2,
        waiting_jobs=(new_job,),
        running_jobs=(existing_running_job,),
        nodes=(_node(available_gpus=6),),
    )

    decision = SchedulingDecision(
        assignments=(
            GpuAssignment(
                job_id="new-job",
                node_id="node-001",
                gpu_count=1,
            ),
        ),
        started_job_ids=("new-job",),
        waiting_job_ids=(),
        objective_value=10.0,
        solver_status="manual",
        runtime_ms=0.0,
    )

    next_state = apply_scheduling_decision(state, decision)

    assert [running.job.job_id for running in next_state.running_jobs] == [
        "already-running",
        "new-job",
    ]
    assert next_state.nodes[0].available_gpus == 5


def test_advance_time_releases_completed_jobs_and_capacity() -> None:
    job = _job("job-a", gpu_demand=2, duration=5)

    running_job = RunningJob(
        job=job,
        assignments=(
            GpuAssignment(
                job_id="job-a",
                node_id="node-001",
                gpu_count=2,
            ),
        ),
        start_time=0,
        completion_time=5,
    )

    state = SimulationState(
        current_time=0,
        waiting_jobs=(),
        running_jobs=(running_job,),
        nodes=(_node(available_gpus=6),),
    )

    next_state = advance_time(state, next_time=5)

    assert next_state.current_time == 5
    assert next_state.running_jobs == ()
    assert [job.job.job_id for job in next_state.completed_jobs] == ["job-a"]
    assert next_state.nodes[0].available_gpus == 8


def test_advance_time_keeps_unfinished_running_jobs() -> None:
    job = _job("job-a", gpu_demand=2, duration=5)

    running_job = RunningJob(
        job=job,
        assignments=(
            GpuAssignment(
                job_id="job-a",
                node_id="node-001",
                gpu_count=2,
            ),
        ),
        start_time=0,
        completion_time=5,
    )

    state = SimulationState(
        current_time=0,
        waiting_jobs=(),
        running_jobs=(running_job,),
        nodes=(_node(available_gpus=6),),
    )

    next_state = advance_time(state, next_time=3)

    assert next_state.current_time == 3
    assert [running.job.job_id for running in next_state.running_jobs] == ["job-a"]
    assert next_state.completed_jobs == ()
    assert next_state.nodes[0].available_gpus == 6


def test_advance_time_rejects_non_increasing_time() -> None:
    state = SimulationState(
        current_time=3,
        waiting_jobs=(),
        running_jobs=(),
        nodes=(_node(),),
    )

    with pytest.raises(ValueError, match="next_time"):
        advance_time(state, next_time=3)


def test_add_arrivals_appends_released_jobs_to_waiting_queue() -> None:
    existing_job = _job("existing-job", gpu_demand=1)
    arriving_job = _job("arriving-job", gpu_demand=2, release_time=3)

    state = SimulationState(
        current_time=3,
        waiting_jobs=(existing_job,),
        running_jobs=(),
        nodes=(_node(),),
    )

    next_state = add_arrivals(state, arriving_jobs=(arriving_job,))

    assert [job.job_id for job in next_state.waiting_jobs] == [
        "existing-job",
        "arriving-job",
    ]


def test_add_arrivals_rejects_future_release_time() -> None:
    arriving_job = _job("future-job", gpu_demand=2, release_time=4)

    state = SimulationState(
        current_time=3,
        waiting_jobs=(),
        running_jobs=(),
        nodes=(_node(),),
    )

    with pytest.raises(ValueError, match="release_time"):
        add_arrivals(state, arriving_jobs=(arriving_job,))
