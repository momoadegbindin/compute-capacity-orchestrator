"""Snapshot scheduling dashboard view."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from compute_capacity_orchestrator.engines.greedy import GreedyScheduler
from compute_capacity_orchestrator.engines.pyomo_snapshot import PyomoSnapshotScheduler
from compute_capacity_orchestrator.engines.validation import validate_decision
from compute_capacity_orchestrator.experiments.scenarios import (
    build_snapshot_with_capacity,
    build_random_snapshot,
)
from compute_capacity_orchestrator.metrics.decision_metrics import (
    compute_decision_metrics,
)

from .snapshot_components import (
    build_assignments_dataframe,
    build_capacity_dataframe,
    build_jobs_dataframe,
    build_nodes_dataframe,
    build_run_history_row,
    render_capacity_chart,
    render_metric_cards,
    render_run_history,
    render_small_snapshot_tables,
    render_large_snapshot_summary
)


AVAILABLE_SCHEDULERS = (
    "Greedy value density",
    "Exact MIP snapshot",
)

PLANNED_SCHEDULERS = (
    "Hybrid bounded-time scheduler",
    "Local search repair",
    "Adaptive large neighborhood search",
    "Column generation",
    "Robust optimization",
    "Stochastic model with recourse",
)
SMALL_EXPERIMENT = "Small"
LARGE_EXPERIMENT = "Large"

MAX_LARGE_NODES = 100
MAX_LARGE_JOBS = 500

DEFAULT_AVAILABLE_CAPACITY_RATIO = 0.70
DEFAULT_TOTAL_GPUS_PER_NODE = 8

def build_scheduler(
    scheduler_name: str,
    deadline_penalty_weight: float,
    time_limit_seconds: float | None = None,
    relative_gap: float | None = None,
):
    """Build a scheduler from the selected dashboard option."""

    if scheduler_name == "Greedy value density":
        return GreedyScheduler()

    if scheduler_name == "Exact MIP snapshot":
        return PyomoSnapshotScheduler(
            deadline_penalty_weight=deadline_penalty_weight,
            decision_step=1,
            time_limit_seconds=time_limit_seconds,
            relative_gap=relative_gap,
            log_diagnostics= False,
        )

    raise ValueError(f"Unknown scheduler: {scheduler_name}")

def render_snapshot_view() -> None:
    """Render the snapshot scheduling dashboard."""

    with st.sidebar:
        st.header("Scenario controls")
        st.caption("Snapshot scheduling")

        experiment_size = st.selectbox(
            "Experiment size",
            options=(SMALL_EXPERIMENT, LARGE_EXPERIMENT),
        )
        scheduler_name = st.selectbox(
            "Available scheduler",
            options=AVAILABLE_SCHEDULERS,
        )
        with st.form("scenario_form"):


            deadline_penalty_weight = st.slider(
                "Deadline penalty weight",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
            )

            if scheduler_name == "Exact MIP snapshot":
                st.subheader("Exact MIP limits")

                mip_col_1, mip_col_2 = st.columns(2)

                with mip_col_1:
                    time_limit_seconds = st.number_input(
                        "Time limit sec",
                        min_value=1,
                        max_value=120,
                        value=15,
                        step=1,
                    )

                with mip_col_2:
                    relative_gap = st.number_input(
                        "MIP gap",
                        min_value=0.00,
                        max_value=1.00,
                        value=0.01,
                        step=0.01,
                        format="%.3f",
                    )
            else:
                time_limit_seconds = None
                relative_gap = None

            if experiment_size == SMALL_EXPERIMENT:
                st.subheader("Node capacity")

                node_col_1, node_col_2 = st.columns(2)

                with node_col_1:
                    node_a_available = st.number_input(
                        "node-a GPUs",
                        min_value=0,
                        max_value=8,
                        value=5,
                        step=1,
                    )

                with node_col_2:
                    node_b_available = st.number_input(
                        "node-b GPUs",
                        min_value=0,
                        max_value=8,
                        value=3,
                        step=1,
                    )

                num_nodes = None
                num_jobs = None
                seed = None
                gpu_demand_min = None
                gpu_demand_max = None

            else:
                st.subheader("Workload")

                seed = st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=100_000,
                    value=27,
                    step=1,
                )

                size_col_1, size_col_2 = st.columns(2)

                with size_col_1:
                    num_nodes = st.number_input(
                        "Nodes",
                        min_value=1,
                        max_value=MAX_LARGE_NODES,
                        value=20,
                        step=1,
                    )

                with size_col_2:
                    num_jobs = st.number_input(
                        "Jobs",
                        min_value=1,
                        max_value=MAX_LARGE_JOBS,
                        value=100,
                        step=10,
                    )

                gpu_col_1, gpu_col_2 = st.columns(2)

                with gpu_col_1:
                    gpu_demand_min = st.number_input(
                        "Min GPUs",
                        min_value=1,
                        max_value=64,
                        value=1,
                        step=1,
                    )

                with gpu_col_2:
                    gpu_demand_max = st.number_input(
                        "Max GPUs",
                        min_value=int(gpu_demand_min),
                        max_value=64,
                        value=max(int(gpu_demand_min), 16),
                        step=1,
                    )

                node_a_available = None
                node_b_available = None

            run_requested = st.form_submit_button("Run scheduler")

        st.caption(
            "Later versions will add simulation, time-indexed planning, "
            "and policy comparison."
        )

        st.divider()
        st.caption("Planned engines")

        for planned_scheduler in PLANNED_SCHEDULERS:
            st.markdown(
                f"<span style='color: #8b949e;'>• {planned_scheduler}</span>",
                unsafe_allow_html=True,
            )

    if "run_history" not in st.session_state:
        st.session_state["run_history"] = []

    if "snapshot_last_result" not in st.session_state:
        st.session_state["snapshot_last_result"] = None

    if run_requested:
        try:
            if experiment_size == SMALL_EXPERIMENT:
                snapshot = build_snapshot_with_capacity(
                    node_a_available=int(node_a_available),
                    node_b_available=int(node_b_available),
                )
                scenario_name = "small-capacity"

            else:
                snapshot = build_random_snapshot(
                    num_nodes=int(num_nodes),
                    num_jobs=int(num_jobs),
                    seed=int(seed),
                    gpu_demand_range=(int(gpu_demand_min), int(gpu_demand_max)),
                    duration_range=(1, 24),
                    priority_range=(1.0, 100.0),
                    deadline_slack_range=(-4, 24),
                    total_gpus_per_node=DEFAULT_TOTAL_GPUS_PER_NODE,
                    available_capacity_ratio=DEFAULT_AVAILABLE_CAPACITY_RATIO,
                )
                scenario_name = f"large-generated-{int(num_nodes)}x{int(num_jobs)}"

            scheduler = build_scheduler(
                scheduler_name=scheduler_name,
                deadline_penalty_weight=deadline_penalty_weight,
                time_limit_seconds=time_limit_seconds,
                relative_gap=relative_gap,
            )

            decision = scheduler.solve(snapshot)
            validate_decision(snapshot, decision)

            metrics = compute_decision_metrics(
                snapshot=snapshot,
                decision=decision,
                deadline_penalty_weight=deadline_penalty_weight,
                decision_step=1,
            )

            st.session_state["snapshot_last_result"] = {
                "snapshot": snapshot,
                "decision": decision,
                "metrics": metrics,
                "scenario_name": scenario_name,
                "experiment_size": experiment_size,
            }

            st.session_state["run_history"].append(
                build_run_history_row(
                    snapshot=snapshot,
                    metrics=metrics,
                    scheduler_name=scheduler_name,
                    scenario_name=scenario_name,
                    time_label=datetime.now().strftime("%H:%M:%S"),
                )
            )


        except Exception as exc:
            st.session_state["snapshot_last_result"] = None
            st.error(f"Scheduler run failed: {exc}")
            st.stop()

    last_result = st.session_state["snapshot_last_result"]

    if last_result is None:
        st.caption("Click Run scheduler to generate the first result.")
        render_run_history()
        return

    snapshot = last_result["snapshot"]
    decision = last_result["decision"]
    metrics = last_result["metrics"]
    experiment_size = last_result["experiment_size"]

    render_metric_cards(metrics)

    st.caption(
        "Validation checks snapshot feasibility: known jobs, known nodes, "
        "full GPU demand, and node capacity."
    )

    st.divider()

    jobs_df = build_jobs_dataframe(snapshot, decision)
    nodes_df = build_nodes_dataframe(snapshot)
    assignments_df = build_assignments_dataframe(decision)
    capacity_df = build_capacity_dataframe(snapshot, assignments_df)

    if experiment_size == SMALL_EXPERIMENT:
        render_small_snapshot_tables(
            jobs_df=jobs_df,
            nodes_df=nodes_df,
            assignments_df=assignments_df,
        )

        render_capacity_chart(capacity_df)


    else:
        render_large_snapshot_summary(
            jobs_df=jobs_df,
            assignments_df=assignments_df,
            capacity_df=capacity_df,
            metrics=metrics,
        )

    st.caption(
        "Each policy returns a SchedulingDecision. The decision is validated "
        "against the snapshot before metrics are displayed."
    )

    render_run_history()