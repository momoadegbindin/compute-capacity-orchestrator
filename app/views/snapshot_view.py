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
from app.views import tooltips as tips
from app.views.input_validation import validate_number, validate_order
from app.views import control_config as cfg


def validate_snapshot_controls(
    *,
    scheduler_name: str,
    experiment_size: str,
    deadline_penalty_weight: float,
    time_limit_seconds: int | float | None,
    relative_gap: float | None,
    node_a_available: int | None = None,
    node_b_available: int | None = None,
    seed: int | None = None,
    num_nodes: int | None = None,
    num_jobs: int | None = None,
    gpu_demand_min: int | None = None,
    gpu_demand_max: int | None = None,
) -> list[str]:
    """Validate snapshot controls before solving."""
    errors: list[str] = []

    errors.extend(
        validate_number(
            name="Deadline penalty weight",
            value=deadline_penalty_weight,
            min_value=cfg.DEADLINE_WEIGHT_MIN,
            max_value=cfg.DEADLINE_WEIGHT_MAX,
        )
    )

    if experiment_size == cfg.SMALL_EXPERIMENT:
        if node_a_available is None:
            errors.append("node-a GPU availability is required.")
        else:
            errors.extend(
                validate_number(
                    name="node-a GPUs",
                    value=node_a_available,
                    min_value=cfg.NODE_AVAILABLE_GPU_MIN,
                    max_value=cfg.NODE_AVAILABLE_GPU_MAX,
                    integer=True,
                )
            )

        if node_b_available is None:
            errors.append("node-b GPU availability is required.")
        else:
            errors.extend(
                validate_number(
                    name="node-b GPUs",
                    value=node_b_available,
                    min_value=cfg.NODE_AVAILABLE_GPU_MIN,
                    max_value=cfg.NODE_AVAILABLE_GPU_MAX,
                    integer=True,
                )
            )

    elif experiment_size == cfg.LARGE_EXPERIMENT:
        if seed is None:
            errors.append("Random seed is required.")
        else:
            errors.extend(
                validate_number(
                    name="Random seed",
                    value=seed,
                    min_value=cfg.SEED_MIN,
                    max_value=cfg.SEED_MAX,
                    integer=True,
                )
            )

        if num_nodes is None:
            errors.append("Number of nodes is required.")
        else:
            errors.extend(
                validate_number(
                    name="Number of nodes",
                    value=num_nodes,
                    min_value=cfg.NODE_MIN,
                    max_value=cfg.NODE_MAX,
                    integer=True,
                )
            )

        if num_jobs is None:
            errors.append("Number of jobs is required.")
        else:
            errors.extend(
                validate_number(
                    name="Number of jobs",
                    value=num_jobs,
                    min_value=cfg.JOB_MIN,
                    max_value=cfg.LARGE_JOB_MAX,
                    integer=True,
                )
            )

        if gpu_demand_min is None:
            errors.append("Minimum GPU demand is required.")
        else:
            errors.extend(
                validate_number(
                    name="Minimum GPU demand",
                    value=gpu_demand_min,
                    min_value=cfg.GPU_DEMAND_MIN_VALUE,
                    max_value=cfg.GPU_DEMAND_MAX_VALUE,
                    integer=True,
                )
            )

        if gpu_demand_max is None:
            errors.append("Maximum GPU demand is required.")
        else:
            errors.extend(
                validate_number(
                    name="Maximum GPU demand",
                    value=gpu_demand_max,
                    min_value=cfg.GPU_DEMAND_MIN_VALUE,
                    max_value=cfg.GPU_DEMAND_MAX_VALUE,
                    integer=True,
                )
            )

        if gpu_demand_min is not None and gpu_demand_max is not None:
            errors.extend(
                validate_order(
                    low_name="Minimum GPU demand",
                    low_value=gpu_demand_min,
                    high_name="maximum GPU demand",
                    high_value=gpu_demand_max,
                )
            )

        if (
            scheduler_name == cfg.EXACT_MIP_SCHEDULER
            and num_jobs is not None
            and int(num_jobs) > cfg.PUBLIC_EXACT_SNAPSHOT_JOB_MAX
        ):
            errors.append(
                "This snapshot is too large for Exact MIP in the public demo. "
                f"Use at most {cfg.PUBLIC_EXACT_SNAPSHOT_JOB_MAX} jobs "
                f"or switch to {cfg.GREEDY_SCHEDULER}."
            )

    else:
        errors.append(f"Unknown experiment size: {experiment_size}.")

    if scheduler_name == cfg.EXACT_MIP_SCHEDULER:
        if time_limit_seconds is None:
            errors.append("Exact MIP time limit is required.")
        else:
            errors.extend(
                validate_number(
                    name="Exact MIP time limit",
                    value=time_limit_seconds,
                    min_value=cfg.MIP_TIME_LIMIT_MIN,
                    max_value=cfg.MIP_TIME_LIMIT_MAX,
                    integer=True,
                )
            )

            if int(time_limit_seconds) > cfg.PUBLIC_EXACT_SNAPSHOT_TIME_LIMIT_MAX:
                errors.append(
                    "The live demo limits Exact MIP snapshot solves to "
                    f"{cfg.PUBLIC_EXACT_SNAPSHOT_TIME_LIMIT_MAX} seconds."
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

    return errors

def build_scheduler(
    scheduler_name: str,
    deadline_penalty_weight: float,
    time_limit_seconds: float | None = None,
    relative_gap: float | None = None,
):
    """Build a scheduler from the selected dashboard option."""

    if scheduler_name == cfg.GREEDY_SCHEDULER:
        return GreedyScheduler()

    if scheduler_name == cfg.EXACT_MIP_SCHEDULER:
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
            options=cfg.EXPERIMENT_SIZES,
            help=tips.EXPERIMENT_SIZE,
        )
        scheduler_name = st.selectbox(
            "Available scheduler",
            options=cfg.AVAILABLE_SCHEDULERS,
            help=tips.SCHEDULER,
        )
        with st.form("scenario_form"):

            deadline_penalty_weight = st.slider(
                "Deadline penalty weight",
                min_value=cfg.DEADLINE_WEIGHT_MIN,
                max_value=cfg.DEADLINE_WEIGHT_MAX,
                value=cfg.SNAPSHOT_DEADLINE_WEIGHT_DEFAULT,
                step=0.5,
                help=tips.DEADLINE_WEIGHT,
            )

            if scheduler_name == cfg.EXACT_MIP_SCHEDULER:
                st.subheader("Exact MIP limits")

                mip_col_1, mip_col_2 = st.columns(2)

                with mip_col_1:
                    time_limit_seconds = st.number_input(
                        "Time limit sec",
                        #min_value=cfg.MIP_TIME_LIMIT_MIN,
                        #max_value=cfg.MIP_TIME_LIMIT_MAX,
                        value=cfg.SNAPSHOT_MIP_TIME_LIMIT_DEFAULT,
                        step=1,
                        help=tips.MIP_TIME_LIMIT,
                    )

                with mip_col_2:
                    relative_gap = st.number_input(
                        "Relative gap",
                        #min_value=cfg.MIP_GAP_MIN,
                        #max_value=cfg.MIP_GAP_MAX,
                        value=cfg.SNAPSHOT_MIP_GAP_DEFAULT,
                        step=0.01,
                        format="%.3f",
                        help=tips.MIP_GAP,
                    )
            else:
                time_limit_seconds = None
                relative_gap = None

            if experiment_size == cfg.SMALL_EXPERIMENT:
                st.subheader("Node capacity")

                node_col_1, node_col_2 = st.columns(2)

                with node_col_1:
                    node_a_available = st.number_input(
                        "node-a GPUs",
                        #min_value=cfg.NODE_AVAILABLE_GPU_MIN,
                        #max_value=cfg.NODE_AVAILABLE_GPU_MAX,
                        value=5,
                        step=1,
                        help=tips.NODE_AVAILABLE_GPUS,
                    )

                with node_col_2:
                    node_b_available = st.number_input(
                        "node-b GPUs",
                        #min_value=cfg.NODE_AVAILABLE_GPU_MIN,
                        #max_value=cfg.NODE_AVAILABLE_GPU_MAX,
                        value=3,
                        step=1,
                        help=tips.NODE_AVAILABLE_GPUS,
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
                    #min_value=cfg.SEED_MIN,
                    #max_value=cfg.SEED_MAX,
                    value=cfg.SNAPSHOT_SEED_DEFAULT,
                    step=1,
                    help=tips.SEED,
                )

                size_col_1, size_col_2 = st.columns(2)

                with size_col_1:
                    num_nodes = st.number_input(
                        "Nodes",
                        #min_value=cfg.NODE_MIN,
                        #max_value=cfg.NODE_MAX,
                        value=20,
                        step=1,
                        help=tips.NUM_NODES,
                    )

                with size_col_2:
                    num_jobs = st.number_input(
                        "Jobs",
                        #min_value=cfg.JOB_MIN,
                        #max_value=cfg.LARGE_JOB_MAX,
                        value=cfg.LARGE_JOB_DEFAULT,
                        step=10,
                        help=tips.NUM_JOBS,
                    )

                gpu_col_1, gpu_col_2 = st.columns(2)

                with gpu_col_1:
                    gpu_demand_min = st.number_input(
                        "Min GPUs",
                        #min_value=cfg.GPU_DEMAND_MIN_VALUE,
                        #max_value=cfg.GPU_DEMAND_MAX_VALUE,
                        value=cfg.GPU_DEMAND_MIN_DEFAULT,
                        step=1,
                        help=tips.GPU_DEMAND_MIN,
                    )

                with gpu_col_2:
                    gpu_demand_max = st.number_input(
                        "Max GPUs",
                        #min_value=cfg.GPU_DEMAND_MIN_VALUE,
                        #max_value=cfg.GPU_DEMAND_MAX_VALUE,
                        value=cfg.GPU_DEMAND_MAX_DEFAULT,
                        step=1,
                        help=tips.GPU_DEMAND_MAX,
                    )

                node_a_available = None
                node_b_available = None

            run_requested = st.form_submit_button("Run scheduler")
            validation_message_box = st.empty()
        st.caption(
            "Later versions will add simulation, time-indexed planning, "
            "and policy comparison."
        )

        st.divider()
        st.caption("Planned engines")

        for planned_scheduler in cfg.PLANNED_SCHEDULERS:
            st.markdown(
                f"<span style='color: #8b949e;'>• {planned_scheduler}</span>",
                unsafe_allow_html=True,
            )

    if cfg.SNAPSHOT_RUN_HISTORY_KEY not in st.session_state:
        st.session_state[cfg.SNAPSHOT_RUN_HISTORY_KEY] = []

    if cfg.SNAPSHOT_LAST_RESULT_KEY not in st.session_state:
        st.session_state[cfg.SNAPSHOT_LAST_RESULT_KEY] = None

    if run_requested:
        validation_message_box.empty()
        validation_errors = validate_snapshot_controls(
            scheduler_name=scheduler_name,
            experiment_size=experiment_size,
            deadline_penalty_weight=float(deadline_penalty_weight),
            time_limit_seconds=time_limit_seconds,
            relative_gap=relative_gap,
            node_a_available=node_a_available,
            node_b_available=node_b_available,
            seed=seed,
            num_nodes=num_nodes,
            num_jobs=num_jobs,
            gpu_demand_min=gpu_demand_min,
            gpu_demand_max=gpu_demand_max,
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
        try:
            if experiment_size == cfg.SMALL_EXPERIMENT:
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
                    total_gpus_per_node=cfg.TOTAL_GPUS_PER_NODE_DEFAULT,
                    available_capacity_ratio=cfg.DEFAULT_AVAILABLE_CAPACITY_RATIO,
                )
                scenario_name = f"large-generated-{int(num_nodes)}x{int(num_jobs)}"

            scheduler = build_scheduler(
                scheduler_name=scheduler_name,
                deadline_penalty_weight=deadline_penalty_weight,
                time_limit_seconds=time_limit_seconds,
                relative_gap=relative_gap,
            )

            with st.spinner("Solving scheduling snapshot..."):
                decision = scheduler.solve(snapshot)
                validate_decision(snapshot, decision)

                metrics = compute_decision_metrics(
                    snapshot=snapshot,
                    decision=decision,
                    deadline_penalty_weight=deadline_penalty_weight,
                    decision_step=1,
                )

            st.session_state[cfg.SNAPSHOT_LAST_RESULT_KEY] = {
                "snapshot": snapshot,
                "decision": decision,
                "metrics": metrics,
                "scenario_name": scenario_name,
                "experiment_size": experiment_size,
            }

            st.session_state[cfg.SNAPSHOT_RUN_HISTORY_KEY].append(
                build_run_history_row(
                    snapshot=snapshot,
                    metrics=metrics,
                    scheduler_name=scheduler_name,
                    scenario_name=scenario_name,
                    time_label=datetime.now().strftime("%H:%M:%S"),
                )
            )
            st.session_state[cfg.SNAPSHOT_RUN_HISTORY_KEY] = (
                st.session_state[cfg.SNAPSHOT_RUN_HISTORY_KEY][-10:]
            )

        except Exception as exc:
            st.session_state[cfg.SNAPSHOT_LAST_RESULT_KEY] = None
            st.error(f"Scheduler run failed: {exc}")
            st.stop()

    last_result = st.session_state[cfg.SNAPSHOT_LAST_RESULT_KEY]

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



    if experiment_size == cfg.SMALL_EXPERIMENT:
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