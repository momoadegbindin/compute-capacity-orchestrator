from __future__ import annotations

import pytest

from compute_capacity_orchestrator.engines.greedy import GreedyScheduler
from compute_capacity_orchestrator.schemas.resources import ClusterNode
from compute_capacity_orchestrator.schemas.workload import JobRequest
from compute_capacity_orchestrator.simulation.simulation_loop import (
    compute_simulation_summary,
    run_deterministic_simulation,
)
from compute_capacity_orchestrator.simulation.state import SimulationState


class FailingScheduler:
    def solve(self, snapshot):  # noqa: ANN001
        raise AssertionError("Scheduler should not be called for an empty queue")


def _job(
    job_id: str,
    gpu_demand: int,
    duration: int,
    priority: float,
    release_time: int,
    deadline: int,
) -> JobRequest:
    return JobRequest(
        job_id=job_id,
        gpu_demand=gpu_demand,
        duration=duration,
        priority=priority,
        release_time=release_time,
        deadline=deadline,
    )


def _initial_state() -> SimulationState:
    return SimulationState(
        current_time=0,
        waiting_jobs=(),
        running_jobs=(),
        nodes=(
            ClusterNode(
                node_id="node-001",
                total_gpus=4,
                available_gpus=4,
            ),
        ),
    )


def test_run_deterministic_simulation_starts_waits_and_completes_jobs() -> None:
    job_a = _job(
        job_id="job-a",
        gpu_demand=2,
        duration=1,
        priority=10.0,
        release_time=0,
        deadline=10,
    )
    job_b = _job(
        job_id="job-b",
        gpu_demand=4,
        duration=2,
        priority=5.0,
        release_time=0,
        deadline=10,
    )

    result = run_deterministic_simulation(
        initial_state=_initial_state(),
        scheduler=GreedyScheduler(),
        arrivals_by_time={
            0: (job_a, job_b),
        },
        horizon=3,
    )

    summary = compute_simulation_summary(result)

    assert result.final_state.current_time == 3
    assert result.final_state.waiting_jobs == ()
    assert result.final_state.running_jobs == ()
    assert [job.job.job_id for job in result.final_state.completed_jobs] == [
        "job-a",
        "job-b",
    ]

    assert [step.jobs_started for step in result.step_metrics] == [1, 1, 0]
    assert [step.jobs_waiting for step in result.step_metrics] == [1, 0, 0]
    assert [step.jobs_completed_this_step for step in result.step_metrics] == [
        1,
        0,
        1,
    ]

    assert summary.total_jobs_arrived == 2
    assert summary.total_jobs_completed == 2
    assert summary.total_priority_completed == 15.0
    assert summary.deadline_miss_count == 0
    assert summary.deadline_miss_rate == 0.0
    assert summary.average_completed_wait_time == 0.5
    assert summary.jobs_waiting_final == 0
    assert summary.jobs_running_final == 0


def test_run_deterministic_simulation_records_gpu_utilization() -> None:
    job_a = _job(
        job_id="job-a",
        gpu_demand=2,
        duration=1,
        priority=10.0,
        release_time=0,
        deadline=10,
    )
    job_b = _job(
        job_id="job-b",
        gpu_demand=4,
        duration=2,
        priority=5.0,
        release_time=0,
        deadline=10,
    )

    result = run_deterministic_simulation(
        initial_state=_initial_state(),
        scheduler=GreedyScheduler(),
        arrivals_by_time={
            0: (job_a, job_b),
        },
        horizon=3,
    )

    summary = compute_simulation_summary(result)

    assert [step.gpu_capacity_used for step in result.step_metrics] == [2, 4, 4]
    assert summary.average_gpu_utilization == pytest.approx((0.5 + 1.0 + 1.0) / 3)


def test_run_deterministic_simulation_does_not_call_scheduler_for_empty_queue() -> None:
    result = run_deterministic_simulation(
        initial_state=_initial_state(),
        scheduler=FailingScheduler(),
        arrivals_by_time={},
        horizon=2,
    )

    summary = compute_simulation_summary(result)

    assert result.final_state.current_time == 2
    assert result.step_metrics[0].scheduler_status == "idle"
    assert result.step_metrics[1].scheduler_status == "idle"
    assert summary.average_gpu_utilization == 0.0
    assert summary.average_queue_length == 0.0
    assert summary.average_scheduler_runtime_ms == 0.0
    assert summary.total_jobs_arrived == 0
    assert summary.total_jobs_completed == 0


def test_compute_simulation_summary_counts_deadline_misses() -> None:
    late_job = _job(
        job_id="late-job",
        gpu_demand=4,
        duration=2,
        priority=5.0,
        release_time=0,
        deadline=1,
    )

    result = run_deterministic_simulation(
        initial_state=_initial_state(),
        scheduler=GreedyScheduler(),
        arrivals_by_time={
            0: (late_job,),
        },
        horizon=3,
    )

    summary = compute_simulation_summary(result)

    assert summary.total_jobs_completed == 1
    assert summary.deadline_miss_count == 1
    assert summary.deadline_miss_rate == 1.0


def test_run_deterministic_simulation_rejects_invalid_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        run_deterministic_simulation(
            initial_state=_initial_state(),
            scheduler=GreedyScheduler(),
            arrivals_by_time={},
            horizon=0,
        )


def test_run_deterministic_simulation_rejects_arrivals_before_initial_time() -> None:
    state = SimulationState(
        current_time=3,
        waiting_jobs=(),
        running_jobs=(),
        nodes=(
            ClusterNode(
                node_id="node-001",
                total_gpus=4,
                available_gpus=4,
            ),
        ),
    )

    job = _job(
        job_id="job-a",
        gpu_demand=1,
        duration=1,
        priority=1.0,
        release_time=0,
        deadline=10,
    )

    with pytest.raises(ValueError, match="earlier than"):
        run_deterministic_simulation(
            initial_state=state,
            scheduler=GreedyScheduler(),
            arrivals_by_time={
                2: (job,),
            },
            horizon=2,
        )