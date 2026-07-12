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

from app.views import tooltips as tips
from app.views.input_validation import validate_number
from app.views import control_config as cfg

def validate_simulation_controls(
    *,
    scheduler_name: str,
    horizon: int,
    num_nodes: int,
    arrival_rate: float,
    seed: int,
    deadline_penalty_weight: float,
    time_limit_seconds: int | float | None,
    relative_gap: float | None,
) -> list[str]:
    """Validate simulation controls before running an experiment."""
    errors: list[str] = []

    errors.extend(
        validate_number(
            name="Simulation horizon",
            value=horizon,
            min_value=cfg.SIMULATION_HORIZON_MIN,
            max_value=cfg.SIMULATION_HORIZON_MAX,
            integer=True,
        )
    )
    errors.extend(
        validate_number(
            name="Number of nodes",
            value=num_nodes,
            min_value=cfg.NODE_MIN,
            max_value=cfg.NODE_MAX,
            integer=True,
        )
    )
    errors.extend(
        validate_number(
            name="Arrival rate",
            value=arrival_rate,
            min_value=cfg.ARRIVAL_RATE_MIN,
            max_value=cfg.ARRIVAL_RATE_MAX,
        )
    )
    errors.extend(
        validate_number(
            name="Random seed",
            value=seed,
            min_value=cfg.SEED_MIN,
            max_value=cfg.SEED_MAX,
            integer=True,
        )
    )
    errors.extend(
        validate_number(
            name="Deadline penalty weight",
            value=deadline_penalty_weight,
            min_value=cfg.DEADLINE_WEIGHT_MIN,
            max_value=cfg.DEADLINE_WEIGHT_MAX,
        )
    )

    expected_arrivals = float(arrival_rate) * int(horizon)

    if expected_arrivals > cfg.PUBLIC_SIMULATION_EXPECTED_ARRIVALS_MAX:
        errors.append(
            "This simulation is too large for the public demo. "
            f"Arrival rate × horizon is {expected_arrivals:.0f}; "
            f"keep it at or below {cfg.PUBLIC_SIMULATION_EXPECTED_ARRIVALS_MAX}."
        )

    if scheduler_name == cfg.EXACT_MIP_SCHEDULER:
        if int(horizon) > cfg.PUBLIC_EXACT_SIMULATION_HORIZON_MAX:
            errors.append(
                "Exact MIP simulation solves one optimization model per step. "
                f"Use horizon <= {cfg.PUBLIC_EXACT_SIMULATION_HORIZON_MAX} "
                f"or switch to {cfg.GREEDY_SCHEDULER}."
            )

        if time_limit_seconds is None:
            errors.append("Exact MIP time limit is required.")
        else:
            errors.extend(
                validate_number(
                    name="Exact MIP time limit per decision",
                    value=time_limit_seconds,
                    min_value=cfg.MIP_TIME_LIMIT_MIN,
                    max_value=cfg.MIP_TIME_LIMIT_MAX,
                    integer=True,
                )
            )

            if int(time_limit_seconds) > cfg.PUBLIC_EXACT_SIMULATION_TIME_LIMIT_MAX:
                errors.append(
                    "The live demo limits Exact MIP to "
                    f"{cfg.PUBLIC_EXACT_SIMULATION_TIME_LIMIT_MAX} seconds per decision."
                )

        if relative_gap is None:
            errors.append("Relative gap is required for Exact MIP.")
        else:
            errors.extend(
                validate_number(
                    name="Relative gap",
                    value=relative_gap,
                    min_value=cfg.MIP_GAP_MIN,
                    max_value=cfg.MIP_GAP_MAX,
                )
            )

        if expected_arrivals > cfg.PUBLIC_EXACT_SIMULATION_EXPECTED_ARRIVALS_MAX:
            errors.append(
                "This Exact MIP simulation is too large for the public demo. "
                f"Arrival rate × horizon is {expected_arrivals:.0f}; "
                f"keep it at or below {cfg.PUBLIC_EXACT_SIMULATION_EXPECTED_ARRIVALS_MAX}."
            )

    return errors


def build_simulation_scheduler(
    scheduler_name: str,
    deadline_penalty_weight: float,
    time_limit_seconds: float | None = None,
    relative_gap: float | None = None,
):
    """Build a scheduler for simulation experiments."""

    if scheduler_name == cfg.GREEDY_SCHEDULER:
        return GreedyScheduler()

    if scheduler_name == cfg.EXACT_MIP_SCHEDULER:
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
            total_gpus=cfg.TOTAL_GPUS_PER_NODE_DEFAULT,
            available_gpus=cfg.TOTAL_GPUS_PER_NODE_DEFAULT,
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

    if cfg.SIMULATION_LAST_RESULT_KEY not in st.session_state:
        st.session_state[cfg.SIMULATION_LAST_RESULT_KEY] = None

    if cfg.SIMULATION_RUN_HISTORY_KEY not in st.session_state:
        st.session_state[cfg.SIMULATION_RUN_HISTORY_KEY] = []
    with st.sidebar:
        st.header("Simulation controls")
        st.caption("Closed-loop scheduling over time")

        scheduler_name = st.selectbox(
            "Available scheduler",
            options=cfg.AVAILABLE_SCHEDULERS,
            key="simulation_scheduler_name",
            help=tips.SCHEDULER,
        )

        with st.form("simulation_form"):
            if scheduler_name == cfg.EXACT_MIP_SCHEDULER:
                st.subheader("Exact MIP limits")

                mip_col_1, mip_col_2 = st.columns(2)

                with mip_col_1:
                    time_limit_seconds = st.number_input(
                        "Time limit sec",
                        #min_value=cfg.MIP_TIME_LIMIT_MIN,
                        #max_value=cfg.MIP_TIME_LIMIT_MAX,
                        value=cfg.SIMULATION_MIP_TIME_LIMIT_DEFAULT,
                        step=1,
                        help=tips.MIP_TIME_LIMIT,
                    )

                with mip_col_2:
                    relative_gap = st.number_input(
                        "Relative gap",
                        #min_value=cfg.MIP_GAP_MIN,
                        #max_value=cfg.MIP_GAP_MAX,
                        value=cfg.SIMULATION_MIP_GAP_DEFAULT,
                        step=0.01,
                        format="%.3f",
                        help=tips.MIP_GAP,
                    )
            else:
                time_limit_seconds = None
                relative_gap = None

            st.subheader("System")

            num_nodes = st.number_input(
                "Nodes",
                #min_value=cfg.NODE_MIN,
                #max_value=cfg.NODE_MAX,
                value=cfg.NODE_DEFAULT,
                step=1,
                help=tips.NUM_NODES,
            )

            st.subheader("Workload")

            horizon = st.number_input(
                "Horizon",
                #min_value=cfg.SIMULATION_HORIZON_MIN,
                #max_value=cfg.SIMULATION_HORIZON_MAX,
                value= cfg.SIMULATION_HORIZON_DEFAULT,
                step=5,
                help=tips.HORIZON,
            )

            arrival_rate = st.number_input(
                "Arrival rate",
                #min_value=cfg.ARRIVAL_RATE_MIN,
                #max_value=cfg.ARRIVAL_RATE_MAX,
                value=cfg.ARRIVAL_RATE_DEFAULT,
                step=cfg.ARRIVAL_RATE_STEP,
                format="%.1f",
                help=tips.ARRIVAL_RATE,
            )

            seed = st.number_input(
                "Random seed",
                #min_value=cfg.SEED_MIN,
                #max_value=cfg.SEED_MAX,
                value=cfg.SIMULATION_SEED_DEFAULT,
                step=1,
                help=tips.SEED,
            )

            deadline_penalty_weight = st.slider(
                "Deadline penalty weight",
                min_value=cfg.DEADLINE_WEIGHT_MIN,
                max_value=cfg.DEADLINE_WEIGHT_MAX,
                value=cfg.SIMULATION_DEADLINE_WEIGHT_DEFAULT,
                step=0.5,
                help=tips.DEADLINE_WEIGHT,
            )

            run_requested = st.form_submit_button("Run simulation")
            validation_message_box = st.empty()
    if run_requested:
        validation_message_box.empty()
        try:
            validation_errors = validate_simulation_controls(
                scheduler_name=scheduler_name,
                horizon=int(horizon),
                num_nodes=int(num_nodes),
                arrival_rate=float(arrival_rate),
                seed=int(seed),
                deadline_penalty_weight=float(deadline_penalty_weight),
                time_limit_seconds=time_limit_seconds,
                relative_gap=relative_gap,
            )

            if validation_errors:
                with validation_message_box.container():
                    # Build a single clean markdown list
                    error_bullet_points = "\n".join(f"- {error}" for error in validation_errors)
                    st.error(
                        "### Please correct the following controls before running:\n"
                        f"{error_bullet_points}"
                    )
                return

            with st.spinner("Running scheduling simulation..."):
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

            st.session_state[cfg.SIMULATION_LAST_RESULT_KEY] = {
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

            st.session_state[cfg.SIMULATION_RUN_HISTORY_KEY].append(history_row)
            st.session_state[cfg.SIMULATION_RUN_HISTORY_KEY] = (
                st.session_state[cfg.SIMULATION_RUN_HISTORY_KEY][-10:]
            )
        except Exception as exc:
            st.session_state[cfg.SIMULATION_LAST_RESULT_KEY] = None
            st.error(f"Scheduler run failed: {exc}")
            st.stop()

    last_result = st.session_state[cfg.SIMULATION_LAST_RESULT_KEY]

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

    run_history = st.session_state[cfg.SIMULATION_RUN_HISTORY_KEY]

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