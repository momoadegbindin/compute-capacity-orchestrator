"""Simulation dashboard view."""

from __future__ import annotations

import streamlit as st

from datetime import datetime

import pandas as pd

from compute_capacity_orchestrator.engines.greedy import GreedyScheduler
from compute_capacity_orchestrator.engines.pyomo_snapshot import PyomoSnapshotScheduler
from compute_capacity_orchestrator.schemas.resources import ClusterNode
from compute_capacity_orchestrator.simulation.arrivals import (
    generate_heavy_tailed_arrivals_by_time,
)
from compute_capacity_orchestrator.simulation.simulation_loop import (
    SimulationResult,
    SimulationSummary,
    compute_simulation_summary,
    run_deterministic_simulation,
)
from compute_capacity_orchestrator.simulation.state import SimulationState

from app.views.simulation_components import (
    build_simulation_diagnostics_dataframe,
    build_simulation_step_dataframe,
    build_simulation_run_history_row,
)

AVAILABLE_SCHEDULERS = (
    "Greedy value density",
    "Exact MIP snapshot",
)

SIMULATION_LAST_RESULT_KEY = "simulation_last_result"
SIMULATION_RUN_HISTORY_KEY = "simulation_run_history"

DEFAULT_TOTAL_GPUS_PER_NODE = 8


def build_simulation_scheduler(
    scheduler_name: str,
    deadline_penalty_weight: float,
    time_limit_seconds: float | None = None,
    relative_gap: float | None = None,
):
    """Build a scheduler for simulation experiments."""

    if scheduler_name == "Greedy value density":
        return GreedyScheduler()

    if scheduler_name == "Exact MIP snapshot":
        return PyomoSnapshotScheduler(
            deadline_penalty_weight=deadline_penalty_weight,
            decision_step=1,
            time_limit_seconds=time_limit_seconds,
            relative_gap=relative_gap,
            log_diagnostics=False,
        )

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def build_initial_simulation_state(
    num_nodes: int,
) -> SimulationState:
    """Build the initial empty cluster state for a simulation run."""

    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    nodes = tuple(
        ClusterNode(
            node_id=f"node-{node_index:03d}",
            total_gpus=DEFAULT_TOTAL_GPUS_PER_NODE,
            available_gpus=DEFAULT_TOTAL_GPUS_PER_NODE,
            topology_group=f"rack-{node_index // 4}",
        )
        for node_index in range(num_nodes)
    )

    return SimulationState(
        current_time=0,
        waiting_jobs=(),
        running_jobs=(),
        nodes=nodes,
        completed_jobs=(),
    )


def run_simulation_from_controls(
    scheduler_name: str,
    num_nodes: int,
    horizon: int,
    arrival_rate: float,
    seed: int,
    deadline_penalty_weight: float,
    time_limit_seconds: float | None,
    relative_gap: float | None,
) -> tuple[SimulationResult, SimulationSummary]:
    """Run one simulation from dashboard controls."""

    initial_state = build_initial_simulation_state(
        num_nodes=num_nodes,
    )

    arrivals_by_time = generate_heavy_tailed_arrivals_by_time(
        start_time=0,
        horizon=horizon,
        seed=seed,
        arrival_rate=arrival_rate,
        gpu_demand_range=(1, 16),
        duration_range=(1, 12),
        priority_range=(1.0, 100.0),
        deadline_slack_range=(-2, 12),
        max_arrivals_per_step=12,
    )

    scheduler = build_simulation_scheduler(
        scheduler_name=scheduler_name,
        deadline_penalty_weight=deadline_penalty_weight,
        time_limit_seconds=time_limit_seconds,
        relative_gap=relative_gap,
    )

    result = run_deterministic_simulation(
        initial_state=initial_state,
        scheduler=scheduler,
        arrivals_by_time=arrivals_by_time,
        horizon=horizon,
        deadline_penalty_weight=deadline_penalty_weight,
        decision_step=1,
    )

    summary = compute_simulation_summary(result)

    return result, summary


def render_simulation_view() -> None:
    """Render the simulation dashboard shell."""

    if SIMULATION_LAST_RESULT_KEY not in st.session_state:
        st.session_state[SIMULATION_LAST_RESULT_KEY] = None

    if SIMULATION_RUN_HISTORY_KEY not in st.session_state:
        st.session_state[SIMULATION_RUN_HISTORY_KEY] = []
    with st.sidebar:
        st.header("Simulation controls")
        st.caption("Closed-loop scheduling over time")

        scheduler_name = st.selectbox(
            "Available scheduler",
            options=AVAILABLE_SCHEDULERS,
            key="simulation_scheduler_name",
        )

        with st.form("simulation_form"):
            if scheduler_name == "Exact MIP snapshot":
                st.subheader("Exact MIP limits")

                mip_col_1, mip_col_2 = st.columns(2)

                with mip_col_1:
                    time_limit_seconds = st.number_input(
                        "Time limit sec",
                        min_value=1,
                        max_value=30,
                        value=5,
                        step=1,
                    )

                with mip_col_2:
                    relative_gap = st.number_input(
                        "MIP gap",
                        min_value=0.00,
                        max_value=0.10,
                        value=0.05,
                        step=0.01,
                        format="%.3f",
                    )
            else:
                time_limit_seconds = None
                relative_gap = None

            st.subheader("System")

            num_nodes = st.number_input(
                "Nodes",
                min_value=1,
                max_value=100,
                value=8,
                step=1,
            )

            st.subheader("Workload")

            horizon = st.number_input(
                "Horizon",
                min_value=1,
                max_value=200,
                value=50,
                step=10,
            )

            arrival_rate = st.number_input(
                "Arrival rate",
                min_value=0.0,
                max_value=20.0,
                value=3.0,
                step=0.5,
                format="%.1f",
            )

            seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=100_000,
                value=42,
                step=1,
            )

            deadline_penalty_weight = st.slider(
                "Deadline penalty weight",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
            )

            run_requested = st.form_submit_button("Run simulation")

    if run_requested:
        try:
            result, summary = run_simulation_from_controls(
                scheduler_name=scheduler_name,
                num_nodes=int(num_nodes),
                horizon=int(horizon),
                arrival_rate=float(arrival_rate),
                seed=int(seed),
                deadline_penalty_weight=float(deadline_penalty_weight),
                time_limit_seconds=(
                    float(time_limit_seconds)
                    if time_limit_seconds is not None
                    else None
                ),
                relative_gap=(
                    float(relative_gap)
                    if relative_gap is not None
                    else None
                ),
            )

            st.session_state[SIMULATION_LAST_RESULT_KEY] = {
                "result": result,
                "summary": summary,
                "scheduler_name": scheduler_name,
                "num_nodes": int(num_nodes),
                "horizon": int(horizon),
                "arrival_rate": float(arrival_rate),
                "seed": int(seed),
            }
            history_row = build_simulation_run_history_row(
                summary=summary,
                scheduler_name=scheduler_name,
                num_nodes=int(num_nodes),
                horizon=int(horizon),
                arrival_rate=float(arrival_rate),
                seed=int(seed),
                time_label=datetime.now().strftime("%H:%M:%S"),
            )

            st.session_state[SIMULATION_RUN_HISTORY_KEY].append(history_row)
            st.session_state[SIMULATION_RUN_HISTORY_KEY] = (
                st.session_state[SIMULATION_RUN_HISTORY_KEY][-10:]
            )
        except Exception as exc:
            st.session_state[SIMULATION_LAST_RESULT_KEY] = None
            st.error(f"Scheduler run failed: {exc}")
            st.stop()

    last_result = st.session_state[SIMULATION_LAST_RESULT_KEY]

    if last_result is None:
        st.caption("Click Run simulation to generate the first result.")
        return

    result = last_result["result"]
    summary = last_result["summary"]

    step_df = build_simulation_step_dataframe(result)
    diagnostics_df = build_simulation_diagnostics_dataframe(summary)

    st.subheader("Last simulation result")

    st.caption(
        "Result from: "
        f"{last_result['scheduler_name']} | "
        f"{last_result['num_nodes']} nodes | "
        f"horizon {last_result['horizon']} | "
        f"arrival rate {last_result['arrival_rate']} | "
        f"seed {last_result['seed']}"
    )

    metric_cols = st.columns(4)

    metric_cols[0].metric(
        "Completed / arrived",
        f"{summary.total_jobs_completed}/{summary.total_jobs_arrived}",
    )
    metric_cols[1].metric(
        "Final waiting",
        summary.jobs_waiting_final,
    )
    metric_cols[2].metric(
        "Final running",
        summary.jobs_running_final,
    )
    metric_cols[3].metric(
        "Avg GPU utilization",
        f"{summary.average_gpu_utilization:.1%}",
    )

    metric_cols = st.columns(4)

    metric_cols[0].metric(
        "Avg queue length",
        f"{summary.average_queue_length:.2f}",
    )
    metric_cols[1].metric(
        "Avg wait",
        f"{summary.average_completed_wait_time:.2f}",
    )
    metric_cols[2].metric(
        "Deadline miss rate",
        f"{summary.deadline_miss_rate:.1%}",
    )
    metric_cols[3].metric(
        "Avg scheduler runtime",
        f"{summary.average_scheduler_runtime_ms:.2f} ms",
    )

    st.divider()

    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        st.subheader("GPU utilization over time")

        utilization_df = step_df[["time", "gpu_utilization"]].copy()
        utilization_df["GPU utilization (%)"] = (
                utilization_df["gpu_utilization"] * 100.0
        )

        st.line_chart(
            utilization_df.set_index("time")[["GPU utilization (%)"]],
        )

    with chart_col_2:
        st.subheader("Queue length over time")

        st.line_chart(
            step_df.set_index("time")[["jobs_waiting"]],
        )

    st.subheader("Simulation diagnostics")

    st.dataframe(
        diagnostics_df,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Step metrics", expanded=False):
        st.dataframe(
            step_df,
            use_container_width=True,
            hide_index=True,
        )

    run_history = st.session_state[SIMULATION_RUN_HISTORY_KEY]

    if run_history:
        with st.expander("Recent simulation runs", expanded=False):
            st.dataframe(
                pd.DataFrame(reversed(run_history)),
                use_container_width=True,
                hide_index=True,
            )
    st.caption(
        "Changing controls does not update this result. "
        "Click Run simulation to generate a new result."
    )