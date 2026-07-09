from __future__ import annotations

import pytest

from compute_capacity_orchestrator.schemas.resources import ClusterNode
from compute_capacity_orchestrator.schemas.schedule import GpuAssignment
from compute_capacity_orchestrator.schemas.workload import JobRequest
from compute_capacity_orchestrator.simulation.state import RunningJob, SimulationState


def test_running_job_accepts_valid_assignment() -> None:
    job = JobRequest(
        job_id="job-001",
        gpu_demand=2,
        duration=5,
        priority=10.0,
        release_time=0,
        deadline=20,
    )

    running_job = RunningJob(
        job=job,
        assignments=(
            GpuAssignment(
                job_id="job-001",
                node_id="node-001",
                gpu_count=2,
            ),
        ),
        start_time=0,
        completion_time=5,
    )

    assert running_job.job.job_id == "job-001"
    assert running_job.completion_time == 5


def test_running_job_rejects_wrong_completion_time() -> None:
    job = JobRequest(
        job_id="job-001",
        gpu_demand=2,
        duration=5,
        priority=10.0,
        release_time=0,
        deadline=20,
    )

    with pytest.raises(ValueError, match="completion_time"):
        RunningJob(
            job=job,
            assignments=(
                GpuAssignment(
                    job_id="job-001",
                    node_id="node-001",
                    gpu_count=2,
                ),
            ),
            start_time=0,
            completion_time=6,
        )


def test_running_job_rejects_assignment_demand_mismatch() -> None:
    job = JobRequest(
        job_id="job-001",
        gpu_demand=2,
        duration=5,
        priority=10.0,
        release_time=0,
        deadline=20,
    )

    with pytest.raises(ValueError, match="assigned_gpus"):
        RunningJob(
            job=job,
            assignments=(
                GpuAssignment(
                    job_id="job-001",
                    node_id="node-001",
                    gpu_count=1,
                ),
            ),
            start_time=0,
            completion_time=5,
        )


def test_simulation_state_accepts_consistent_state() -> None:
    running_job_request = JobRequest(
        job_id="running-job",
        gpu_demand=2,
        duration=5,
        priority=10.0,
        release_time=0,
        deadline=20,
    )

    waiting_job = JobRequest(
        job_id="waiting-job",
        gpu_demand=1,
        duration=3,
        priority=5.0,
        release_time=0,
        deadline=10,
    )

    running_job = RunningJob(
        job=running_job_request,
        assignments=(
            GpuAssignment(
                job_id="running-job",
                node_id="node-001",
                gpu_count=2,
            ),
        ),
        start_time=0,
        completion_time=5,
    )

    state = SimulationState(
        current_time=1,
        waiting_jobs=(waiting_job,),
        running_jobs=(running_job,),
        nodes=(
            ClusterNode(
                node_id="node-001",
                total_gpus=8,
                available_gpus=6,
            ),
        ),
    )

    assert state.current_time == 1
    assert len(state.waiting_jobs) == 1
    assert len(state.running_jobs) == 1


def test_simulation_state_rejects_waiting_running_overlap() -> None:
    job = JobRequest(
        job_id="job-001",
        gpu_demand=2,
        duration=5,
        priority=10.0,
        release_time=0,
        deadline=20,
    )

    running_job = RunningJob(
        job=job,
        assignments=(
            GpuAssignment(
                job_id="job-001",
                node_id="node-001",
                gpu_count=2,
            ),
        ),
        start_time=0,
        completion_time=5,
    )

    with pytest.raises(ValueError, match="both waiting and running"):
        SimulationState(
            current_time=1,
            waiting_jobs=(job,),
            running_jobs=(running_job,),
            nodes=(
                ClusterNode(
                    node_id="node-001",
                    total_gpus=8,
                    available_gpus=6,
                ),
            ),
        )


def test_simulation_state_rejects_unreleased_waiting_job() -> None:
    waiting_job = JobRequest(
        job_id="future-job",
        gpu_demand=1,
        duration=3,
        priority=5.0,
        release_time=2,
        deadline=10,
    )

    with pytest.raises(ValueError, match="release_time"):
        SimulationState(
            current_time=1,
            waiting_jobs=(waiting_job,),
            running_jobs=(),
            nodes=(
                ClusterNode(
                    node_id="node-001",
                    total_gpus=8,
                    available_gpus=8,
                ),
            ),
        )


def test_simulation_state_rejects_node_overallocation() -> None:
    running_job_request = JobRequest(
        job_id="running-job",
        gpu_demand=4,
        duration=5,
        priority=10.0,
        release_time=0,
        deadline=20,
    )

    running_job = RunningJob(
        job=running_job_request,
        assignments=(
            GpuAssignment(
                job_id="running-job",
                node_id="node-001",
                gpu_count=4,
            ),
        ),
        start_time=0,
        completion_time=5,
    )

    with pytest.raises(ValueError, match="cannot exceed total_gpus"):
        SimulationState(
            current_time=1,
            waiting_jobs=(),
            running_jobs=(running_job,),
            nodes=(
                ClusterNode(
                    node_id="node-001",
                    total_gpus=8,
                    available_gpus=5,
                ),
            ),
        )