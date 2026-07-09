from __future__ import annotations

import pytest

from compute_capacity_orchestrator.engines.greedy import GreedyScheduler
from compute_capacity_orchestrator.engines.pyomo_snapshot import PyomoSnapshotScheduler

from app.views.simulation_view import (
    build_initial_simulation_state,
    build_simulation_scheduler,
    run_simulation_from_controls,
)


def test_build_simulation_scheduler_returns_greedy_scheduler() -> None:
    scheduler = build_simulation_scheduler(
        scheduler_name="Greedy value density",
        deadline_penalty_weight=1.0,
    )

    assert isinstance(scheduler, GreedyScheduler)


def test_build_simulation_scheduler_returns_pyomo_scheduler_with_limits() -> None:
    scheduler = build_simulation_scheduler(
        scheduler_name="Exact MIP snapshot",
        deadline_penalty_weight=2.0,
        time_limit_seconds=5,
        relative_gap=0.05,
    )

    assert isinstance(scheduler, PyomoSnapshotScheduler)
    assert scheduler.deadline_penalty_weight == 2.0
    assert scheduler.decision_step == 1
    assert scheduler.time_limit_seconds == 5
    assert scheduler.relative_gap == 0.05
    assert scheduler.log_diagnostics is False


def test_build_simulation_scheduler_rejects_unknown_scheduler() -> None:
    with pytest.raises(ValueError, match="Unknown scheduler"):
        build_simulation_scheduler(
            scheduler_name="unknown",
            deadline_penalty_weight=0.0,
        )


def test_build_initial_simulation_state_creates_empty_cluster() -> None:
    state = build_initial_simulation_state(num_nodes=3)

    assert state.current_time == 0
    assert state.waiting_jobs == ()
    assert state.running_jobs == ()
    assert state.completed_jobs == ()
    assert len(state.nodes) == 3

    assert [node.node_id for node in state.nodes] == [
        "node-000",
        "node-001",
        "node-002",
    ]
    assert all(node.total_gpus == 8 for node in state.nodes)
    assert all(node.available_gpus == 8 for node in state.nodes)


def test_build_initial_simulation_state_rejects_invalid_node_count() -> None:
    with pytest.raises(ValueError, match="num_nodes"):
        build_initial_simulation_state(num_nodes=0)


def test_run_simulation_from_controls_returns_result_and_summary() -> None:
    result, summary = run_simulation_from_controls(
        scheduler_name="Greedy value density",
        num_nodes=4,
        horizon=10,
        arrival_rate=2.0,
        seed=7,
        deadline_penalty_weight=1.0,
        time_limit_seconds=None,
        relative_gap=None,
    )

    assert len(result.step_metrics) == 10
    assert result.final_state.current_time == 10

    assert summary.total_jobs_arrived >= summary.total_jobs_completed
    assert summary.jobs_waiting_final >= 0
    assert summary.jobs_running_final >= 0
    assert 0.0 <= summary.average_gpu_utilization <= 1.0
    assert 0.0 <= summary.deadline_miss_rate <= 1.0
    assert summary.average_scheduler_runtime_ms >= 0.0


def test_run_simulation_from_controls_is_reproducible_for_same_seed() -> None:
    result_a, summary_a = run_simulation_from_controls(
        scheduler_name="Greedy value density",
        num_nodes=4,
        horizon=10,
        arrival_rate=2.0,
        seed=7,
        deadline_penalty_weight=1.0,
        time_limit_seconds=None,
        relative_gap=None,
    )

    result_b, summary_b = run_simulation_from_controls(
        scheduler_name="Greedy value density",
        num_nodes=4,
        horizon=10,
        arrival_rate=2.0,
        seed=7,
        deadline_penalty_weight=1.0,
        time_limit_seconds=None,
        relative_gap=None,
    )

    assert result_a.final_state == result_b.final_state

    assert summary_a.total_jobs_arrived == summary_b.total_jobs_arrived
    assert summary_a.total_jobs_completed == summary_b.total_jobs_completed
    assert summary_a.jobs_waiting_final == summary_b.jobs_waiting_final
    assert summary_a.jobs_running_final == summary_b.jobs_running_final
    assert summary_a.average_gpu_utilization == summary_b.average_gpu_utilization
    assert summary_a.average_queue_length == summary_b.average_queue_length
    assert summary_a.total_priority_completed == summary_b.total_priority_completed
    assert summary_a.deadline_miss_count == summary_b.deadline_miss_count
    assert summary_a.deadline_miss_rate == summary_b.deadline_miss_rate
    assert summary_a.average_completed_wait_time == summary_b.average_completed_wait_time

    assert summary_a.average_scheduler_runtime_ms >= 0.0
    assert summary_b.average_scheduler_runtime_ms >= 0.0