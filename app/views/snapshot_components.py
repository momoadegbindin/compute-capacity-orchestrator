"""Reusable display helpers for the snapshot dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from compute_capacity_orchestrator.metrics.decision_metrics import DecisionMetrics
from compute_capacity_orchestrator.schemas.schedule import (
    SchedulingDecision,
    SchedulingSnapshot,
)

from app.views import control_config as cfg

def build_jobs_dataframe(
    snapshot: SchedulingSnapshot,
    decision: SchedulingDecision,
) -> pd.DataFrame:
    """Build the queued-jobs table with scheduler outcome labels."""

    rows = [
        {
            "job_id": job.job_id,
            "gpu_demand": job.gpu_demand,
            "duration": job.duration,
            "priority": f"{job.priority:.2f}",
            "release_time": job.release_time,
            "deadline": job.deadline,
            "decision": (
                "start" if job.job_id in decision.started_job_ids else "wait"
            ),

        }
        for job in snapshot.queued_jobs
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "job_id",
            "decision",
            "gpu_demand",
            "duration",
            "priority",
            "release_time",
            "deadline",
        ],
    )


def build_nodes_dataframe(snapshot: SchedulingSnapshot) -> pd.DataFrame:
    """Build the cluster-nodes table."""

    rows = [
        {
            "node_id": node.node_id,
            "available_gpus": node.available_gpus,
            "total_gpus": node.total_gpus,
            "topology_group": node.topology_group,
        }
        for node in snapshot.nodes
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "node_id",
            "available_gpus",
            "total_gpus",
            "topology_group",
        ],
    )


def build_assignments_dataframe(decision: SchedulingDecision) -> pd.DataFrame:
    """Build the post-decision assignment table."""

    rows = [
        {
            "job_id": assignment.job_id,
            "node_id": assignment.node_id,
            "gpu_count": assignment.gpu_count,
        }
        for assignment in decision.assignments
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "job_id",
            "node_id",
            "gpu_count",
        ],
    )


def build_capacity_dataframe(
    snapshot: SchedulingSnapshot,
    assignments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build used/unused GPU capacity by node."""

    used_by_node = (
        assignments_df.groupby("node_id")["gpu_count"].sum().to_dict()
        if not assignments_df.empty
        else {}
    )

    rows = [
        {
            "node_id": node.node_id,
            "used_gpus": used_by_node.get(node.node_id, 0),
            "unused_gpus": node.available_gpus - used_by_node.get(node.node_id, 0),
        }
        for node in snapshot.nodes
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "node_id",
            "used_gpus",
            "unused_gpus",
        ],
    )


def build_run_history_row(
    snapshot: SchedulingSnapshot,
    metrics: DecisionMetrics,
    scheduler_name: str,
    scenario_name: str,
    time_label: str,
) -> dict[str, object]:
    """Build one row for the recent-runs table."""

    total_installed_gpus = sum(node.total_gpus for node in snapshot.nodes)

    return {
        "time": time_label,
        "scheduler": scheduler_name,
        "scenario": scenario_name,
        "jobs": f"{metrics.jobs_started}/{metrics.jobs_submitted}",
        "cluster": (
            f"{len(snapshot.nodes)} nodes, "
            f"{metrics.gpu_capacity_available}/{total_installed_gpus} GPUs available"
        ),
        "gpu use": f"{metrics.gpu_capacity_used}/{metrics.gpu_capacity_available}",
        "utilization": f"{metrics.gpu_utilization:.1%}",
        "objective": round(metrics.objective_value, 2),
        "runtime_ms": round(metrics.runtime_ms, 3),
    }


def style_decision_column(column: pd.Series) -> list[str]:
    """Style start/wait labels in the job table."""

    return [
        (
            "background-color: rgba(46, 160, 67, 0.28); "
            "color: #7ee787; font-weight: 600;"
        )
        if value == "start"
        else (
            "background-color: rgba(248, 81, 73, 0.22); "
            "color: #ff7b72; font-weight: 600;"
        )
        for value in column
    ]


def render_metric_cards(
    metrics: DecisionMetrics,
) -> None:
    """Render compact top-level snapshot metrics."""

    metric_cols = st.columns(5)

    metric_cols[0].metric(
        "Jobs started / submitted",
        f"{metrics.jobs_started}/{metrics.jobs_submitted}",
    )
    metric_cols[1].metric(
        "Jobs waiting",
        str(metrics.jobs_waiting),
    )
    metric_cols[2].metric(
        "Available GPU utilization",
        f"{metrics.gpu_utilization:.1%}",
    )
    metric_cols[3].metric(
        "Objective",
        f"{metrics.scheduler_objective_value:,.2f}",
    )
    metric_cols[4].metric(
        "Runtime / decision",
        f"{metrics.runtime_ms:.2f} ms",
    )
def render_small_snapshot_tables(
    jobs_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
) -> None:
    """Render detailed tables for small snapshot experiments."""

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Queued jobs with scheduler outcome")
        styled_jobs_df = jobs_df.style.apply(
            style_decision_column,
            subset=["decision"],
        )
        st.dataframe(styled_jobs_df, use_container_width=True, hide_index=True)

    with right_col:
        st.subheader("Cluster nodes")
        st.dataframe(nodes_df, use_container_width=True, hide_index=True)

    st.subheader("Post-decision GPU assignments")

    if assignments_df.empty:
        st.info("No jobs were started by this policy.")
    else:
        st.dataframe(assignments_df, use_container_width=True, hide_index=True)


def render_capacity_chart(capacity_df: pd.DataFrame) -> None:
    """Render used and unused GPU capacity by node."""

    st.subheader("GPU capacity by node")
    st.bar_chart(
        capacity_df.set_index("node_id")[["used_gpus", "unused_gpus"]],
    )

def render_large_snapshot_summary(
    jobs_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    metrics: DecisionMetrics,
) -> None:
    """Render compact diagnostics for large snapshot experiments."""

    started_jobs_df = jobs_df[jobs_df["decision"] == "start"]
    waiting_jobs_df = jobs_df[jobs_df["decision"] == "wait"]

    started_count = len(started_jobs_df)
    assignment_count = len(assignments_df)

    used_gpus = int(capacity_df["used_gpus"].sum())
    available_gpus = int(
        capacity_df["used_gpus"].sum() + capacity_df["unused_gpus"].sum()
    )

    avg_started_gpu_demand = (
        started_jobs_df["gpu_demand"].mean()
        if not started_jobs_df.empty
        else 0.0
    )
    avg_waiting_gpu_demand = (
        waiting_jobs_df["gpu_demand"].mean()
        if not waiting_jobs_df.empty
        else 0.0
    )

    if waiting_jobs_df.empty:
        highest_priority_waiting = "none"
    else:
        top_waiting_job = waiting_jobs_df.sort_values(
            ["priority", "gpu_demand"],
            ascending=[False, False],
        ).iloc[0]

        priority_value = float(str(top_waiting_job["priority"]).replace(",", ""))
        gpu_demand_value = int(top_waiting_job["gpu_demand"])

        highest_priority_waiting = (
            f"{top_waiting_job['job_id']} | "
            f"priority {priority_value:,.2f} | "
            f"{gpu_demand_value} GPUs"
        )

    if assignments_df.empty:
        split_jobs = 0
        assignments_per_started_job = 0.0
    else:
        assignments_per_job = assignments_df.groupby("job_id").size()
        split_jobs = int((assignments_per_job > 1).sum())
        assignments_per_started_job = (
            assignment_count / started_count
            if started_count > 0
            else 0.0
        )

    diagnostics_df = pd.DataFrame(
        [
            {
                "Diagnostic": "Capacity used",
                "Value": f"{used_gpus}/{available_gpus} GPUs",
            },
            {
                "Diagnostic": "Started priority",
                "Value": f"{metrics.total_priority_started:,.2f}",
            },
            {
                "Diagnostic": "Deadline penalty",
                "Value": f"{metrics.deadline_penalty_incurred:,.2f}",
            },
            {
                "Diagnostic": "Avg GPU demand started",
                "Value": f"{avg_started_gpu_demand:.2f}",
            },
            {
                "Diagnostic": "Avg GPU demand waiting",
                "Value": f"{avg_waiting_gpu_demand:.2f}",
            },
            {
                "Diagnostic": "Highest-priority waiting job",
                "Value": highest_priority_waiting,
            },
            {
                "Diagnostic": "Split jobs",
                "Value": f"{split_jobs}/{started_count} started",
            },
            {
                "Diagnostic": "Assignments per started job",
                "Value": f"{assignments_per_started_job:.2f}",
            },
        ]
    )

    st.subheader("Large run diagnostics")
    st.dataframe(
        diagnostics_df,
        use_container_width=True,
        hide_index=True,
    )

    histogram_df = _build_node_utilization_histogram(capacity_df)

    st.subheader("Node utilization distribution")
    st.bar_chart(
        histogram_df.set_index("Used GPUs per node")["Node count"],
    )

    with st.expander("Top waiting jobs by priority", expanded=False):
        if waiting_jobs_df.empty:
            st.caption("No waiting jobs.")
        else:
            top_waiting_df = (
                waiting_jobs_df.sort_values(
                    ["priority", "gpu_demand"],
                    ascending=[False, False],
                )
                .head(10)[
                    [
                        "job_id",
                        "gpu_demand",
                        "duration",
                        "priority",
                        "deadline",
                    ]
                ]
            )

            st.dataframe(
                top_waiting_df,
                use_container_width=True,
                hide_index=True,
            )


def _build_node_utilization_histogram(
    capacity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build node-count buckets by used GPU count."""

    buckets = {
        "0": 0,
        "1-2": 0,
        "3-4": 0,
        "5-6": 0,
        "7-8": 0,
        "9+": 0,
    }

    for used_gpus in capacity_df["used_gpus"]:
        used_gpus = int(used_gpus)

        if used_gpus == 0:
            buckets["0"] += 1
        elif used_gpus <= 2:
            buckets["1-2"] += 1
        elif used_gpus <= 4:
            buckets["3-4"] += 1
        elif used_gpus <= 6:
            buckets["5-6"] += 1
        elif used_gpus <= 8:
            buckets["7-8"] += 1
        else:
            buckets["9+"] += 1

    return pd.DataFrame(
        [
            {
                "Used GPUs per node": bucket,
                "Node count": count,
            }
            for bucket, count in buckets.items()
        ]
    )

def render_run_history() -> None:
    """Render recent scheduler runs stored in Streamlit session state."""

    st.subheader("Recent scheduler runs")

    if not st.session_state[cfg.SNAPSHOT_RUN_HISTORY_KEY]:
        st.caption("Run one or more schedulers to compare recent decisions.")
        return

    run_history_df = pd.DataFrame(st.session_state[cfg.SNAPSHOT_RUN_HISTORY_KEY])

    st.dataframe(
        run_history_df.tail(10).iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Clear run history"):
        st.session_state[cfg.SNAPSHOT_RUN_HISTORY_KEY] = []
        st.rerun()