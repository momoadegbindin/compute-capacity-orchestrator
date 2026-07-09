"""Reusable display helpers for the simulation dashboard."""

from __future__ import annotations

import pandas as pd

from compute_capacity_orchestrator.simulation.simulation_loop import (
    SimulationResult,
    SimulationSummary,
)


def build_simulation_step_dataframe(
    result: SimulationResult,
) -> pd.DataFrame:
    """Build per-step simulation metrics for charts and inspection."""

    rows = [
        {
            "time": step.time,
            "jobs_arrived": step.jobs_arrived,
            "jobs_started": step.jobs_started,
            "jobs_waiting": step.jobs_waiting,
            "jobs_running": step.jobs_running,
            "jobs_completed": step.jobs_completed,
            "gpu_utilization": step.gpu_utilization,
            "objective_value": step.objective_value,
            "scheduler_runtime_ms": step.scheduler_runtime_ms,
        }
        for step in result.step_metrics
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "time",
            "jobs_arrived",
            "jobs_started",
            "jobs_waiting",
            "jobs_running",
            "jobs_completed",
            "gpu_utilization",
            "objective_value",
            "scheduler_runtime_ms",
        ],
    )


def build_simulation_diagnostics_dataframe(
    summary: SimulationSummary,
) -> pd.DataFrame:
    """Build compact aggregate diagnostics for a simulation run."""

    rows = [
        {
            "Diagnostic": "Jobs arrived",
            "Value": summary.total_jobs_arrived,
        },
        {
            "Diagnostic": "Jobs completed",
            "Value": summary.total_jobs_completed,
        },
        {
            "Diagnostic": "Jobs waiting at end",
            "Value": summary.jobs_waiting_final,
        },
        {
            "Diagnostic": "Jobs running at end",
            "Value": summary.jobs_running_final,
        },
        {
            "Diagnostic": "Average GPU utilization",
            "Value": f"{summary.average_gpu_utilization:.1%}",
        },
        {
            "Diagnostic": "Average queue length",
            "Value": f"{summary.average_queue_length:.2f}",
        },
        {
            "Diagnostic": "Total priority completed",
            "Value": f"{summary.total_priority_completed:.1f}",
        },
        {
            "Diagnostic": "Deadline misses",
            "Value": summary.deadline_miss_count,
        },
        {
            "Diagnostic": "Deadline miss rate",
            "Value": f"{summary.deadline_miss_rate:.1%}",
        },
        {
            "Diagnostic": "Average completed wait",
            "Value": f"{summary.average_completed_wait_time:.2f}",
        },
        {
            "Diagnostic": "Average scheduler runtime",
            "Value": f"{summary.average_scheduler_runtime_ms:.2f} ms",
        },
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Diagnostic",
            "Value",
        ],
    )


def build_simulation_run_history_row(
    summary: SimulationSummary,
    scheduler_name: str,
    num_nodes: int,
    horizon: int,
    arrival_rate: float,
    seed: int,
    time_label: str,
) -> dict[str, object]:
    """Build one row for recent simulation runs."""

    return {
        "time": time_label,
        "scheduler": scheduler_name,
        "nodes": num_nodes,
        "horizon": horizon,
        "arrival_rate": round(arrival_rate, 2),
        "seed": seed,
        "completed/arrived": (
            f"{summary.total_jobs_completed}/{summary.total_jobs_arrived}"
        ),
        "final_waiting": summary.jobs_waiting_final,
        "avg_utilization": f"{summary.average_gpu_utilization:.1%}",
        "miss_rate": f"{summary.deadline_miss_rate:.1%}",
        "avg_runtime_ms": round(summary.average_scheduler_runtime_ms, 2),
    }