from __future__ import annotations

from compute_capacity_orchestrator.schemas.resources import ClusterNode
from compute_capacity_orchestrator.simulation.simulation_loop import (
    SimulationResult,
    SimulationStepMetrics,
    SimulationSummary,
)
from compute_capacity_orchestrator.simulation.state import SimulationState

from app.views.simulation_components import (
    build_simulation_diagnostics_dataframe,
    build_simulation_run_history_row,
    build_simulation_step_dataframe,
)


def _simulation_result() -> SimulationResult:
    return SimulationResult(
        step_metrics=(
            SimulationStepMetrics(
                time=0,
                jobs_arrived=2,
                jobs_started=1,
                jobs_waiting=1,
                jobs_running=1,
                jobs_completed=0,
                jobs_completed_this_step=0,
                gpu_capacity_total=8,
                gpu_capacity_available=6,
                gpu_capacity_used=2,
                gpu_utilization=0.25,
                total_priority_started=10.0,
                deadline_penalty_incurred=0.0,
                objective_value=10.0,
                scheduler_status="heuristic",
                scheduler_runtime_ms=0.123,
            ),
            SimulationStepMetrics(
                time=1,
                jobs_arrived=1,
                jobs_started=2,
                jobs_waiting=0,
                jobs_running=3,
                jobs_completed=1,
                jobs_completed_this_step=1,
                gpu_capacity_total=8,
                gpu_capacity_available=1,
                gpu_capacity_used=7,
                gpu_utilization=0.875,
                total_priority_started=20.0,
                deadline_penalty_incurred=1.0,
                objective_value=19.0,
                scheduler_status="heuristic",
                scheduler_runtime_ms=0.456,
            ),
        ),
        final_state=SimulationState(
            current_time=2,
            waiting_jobs=(),
            running_jobs=(),
            nodes=(
                ClusterNode(
                    node_id="node-001",
                    total_gpus=8,
                    available_gpus=8,
                ),
            ),
            completed_jobs=(),
        ),
    )


def _simulation_summary() -> SimulationSummary:
    return SimulationSummary(
        total_jobs_arrived=3,
        total_jobs_completed=1,
        jobs_waiting_final=0,
        jobs_running_final=2,
        average_gpu_utilization=0.5625,
        average_queue_length=0.5,
        average_scheduler_runtime_ms=0.29,
        total_priority_completed=30.0,
        deadline_miss_count=1,
        deadline_miss_rate=1.0,
        average_completed_wait_time=0.75,
    )


def test_build_simulation_step_dataframe_contains_expected_columns_and_rows() -> None:
    step_df = build_simulation_step_dataframe(_simulation_result())

    assert list(step_df.columns) == [
        "time",
        "jobs_arrived",
        "jobs_started",
        "jobs_waiting",
        "jobs_running",
        "jobs_completed",
        "gpu_utilization",
        "objective_value",
        "scheduler_runtime_ms",
    ]

    assert len(step_df) == 2
    assert step_df.loc[0, "time"] == 0
    assert step_df.loc[0, "jobs_arrived"] == 2
    assert step_df.loc[1, "jobs_running"] == 3
    assert step_df.loc[1, "gpu_utilization"] == 0.875
    assert step_df.loc[1, "scheduler_runtime_ms"] == 0.456


def test_build_simulation_diagnostics_dataframe_formats_summary_values() -> None:
    diagnostics_df = build_simulation_diagnostics_dataframe(_simulation_summary())

    diagnostics = dict(
        zip(
            diagnostics_df["Diagnostic"],
            diagnostics_df["Value"],
        )
    )

    assert diagnostics["Jobs arrived"] == 3
    assert diagnostics["Jobs completed"] == 1
    assert diagnostics["Jobs waiting at end"] == 0
    assert diagnostics["Jobs running at end"] == 2
    assert diagnostics["Average GPU utilization"] == "56.2%"
    assert diagnostics["Average queue length"] == "0.50"
    assert diagnostics["Total priority completed"] == "30.0"
    assert diagnostics["Deadline misses"] == 1
    assert diagnostics["Deadline miss rate"] == "100.0%"
    assert diagnostics["Average completed wait"] == "0.75"
    assert diagnostics["Average scheduler runtime"] == "0.29 ms"


def test_build_simulation_run_history_row_uses_summary_and_controls() -> None:
    row = build_simulation_run_history_row(
        summary=_simulation_summary(),
        scheduler_name="Greedy value density",
        num_nodes=8,
        horizon=50,
        arrival_rate=3.0,
        seed=42,
        time_label="12:00:00",
    )

    assert row["time"] == "12:00:00"
    assert row["scheduler"] == "Greedy value density"
    assert row["nodes"] == 8
    assert row["horizon"] == 50
    assert row["arrival_rate"] == 3.0
    assert row["seed"] == 42
    assert row["completed/arrived"] == "1/3"
    assert row["final_waiting"] == 0
    assert row["avg_utilization"] == "56.2%"
    assert row["miss_rate"] == "100.0%"
    assert row["avg_runtime_ms"] == 0.29