"""Deterministic simulation loop for scheduling policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from compute_capacity_orchestrator.engines.base import Scheduler
from compute_capacity_orchestrator.metrics.decision_metrics import (
    compute_decision_metrics,
)
from compute_capacity_orchestrator.schemas.schedule import SchedulingDecision
from compute_capacity_orchestrator.schemas.workload import JobRequest
from compute_capacity_orchestrator.simulation.state import RunningJob, SimulationState
from compute_capacity_orchestrator.simulation.transition import (
    add_arrivals,
    advance_time,
    apply_scheduling_decision,
    build_scheduling_snapshot,
)


@dataclass(frozen=True, slots=True)
class SimulationStepMetrics:
    """Metrics recorded for one simulation decision epoch."""

    time: int
    jobs_arrived: int
    jobs_started: int
    jobs_waiting: int
    jobs_running: int
    jobs_completed: int
    jobs_completed_this_step: int
    gpu_capacity_total: int
    gpu_capacity_available: int
    gpu_capacity_used: int
    gpu_utilization: float
    total_priority_started: float
    deadline_penalty_incurred: float
    objective_value: float
    scheduler_status: str
    scheduler_runtime_ms: float


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Raw output of a deterministic simulation run."""

    step_metrics: tuple[SimulationStepMetrics, ...]
    final_state: SimulationState


@dataclass(frozen=True, slots=True)
class SimulationSummary:
    """Aggregate metrics for a completed simulation run."""

    total_jobs_arrived: int
    total_jobs_completed: int
    jobs_waiting_final: int
    jobs_running_final: int
    average_gpu_utilization: float
    average_queue_length: float
    average_scheduler_runtime_ms: float
    total_priority_completed: float
    deadline_miss_count: int
    deadline_miss_rate: float
    average_completed_wait_time: float


def run_deterministic_simulation(
    initial_state: SimulationState,
    scheduler: Scheduler,
    arrivals_by_time: Mapping[int, tuple[JobRequest, ...]],
    horizon: int,
    deadline_penalty_weight: float = 0.0,
    decision_step: int = 1,
) -> SimulationResult:
    """Run a deterministic discrete-time scheduling simulation."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if deadline_penalty_weight < 0:
        raise ValueError("deadline_penalty_weight cannot be negative")

    if decision_step <= 0:
        raise ValueError("decision_step must be positive")

    for arrival_time in arrivals_by_time:
        if arrival_time < initial_state.current_time:
            raise ValueError(
                "arrivals_by_time cannot contain times earlier than "
                "initial_state.current_time"
            )

    state = initial_state
    step_metrics: list[SimulationStepMetrics] = []

    for _ in range(horizon):
        current_time = state.current_time
        arriving_jobs = tuple(arrivals_by_time.get(current_time, ()))

        if arriving_jobs:
            state = add_arrivals(state, arriving_jobs)

        snapshot = build_scheduling_snapshot(state)

        if snapshot.queued_jobs:
            decision = scheduler.solve(snapshot)
        else:
            decision = _build_idle_decision()

        decision_metrics = compute_decision_metrics(
            snapshot=snapshot,
            decision=decision,
            deadline_penalty_weight=deadline_penalty_weight,
            decision_step=decision_step,
        )

        state_after_decision = apply_scheduling_decision(
            state=state,
            decision=decision,
        )

        next_time = current_time + 1
        completed_before_advance = len(state_after_decision.completed_jobs)

        next_state = advance_time(
            state=state_after_decision,
            next_time=next_time,
        )

        jobs_completed_this_step = (
            len(next_state.completed_jobs) - completed_before_advance
        )

        step_metrics.append(
            _build_step_metrics(
                time=current_time,
                jobs_arrived=len(arriving_jobs),
                decision=decision,
                decision_objective_value=decision_metrics.objective_value,
                total_priority_started=decision_metrics.total_priority_started,
                deadline_penalty_incurred=(
                    decision_metrics.deadline_penalty_incurred
                ),
                state_after_decision=state_after_decision,
                jobs_completed=len(next_state.completed_jobs),
                jobs_completed_this_step=jobs_completed_this_step,
            )
        )

        state = next_state

    return SimulationResult(
        step_metrics=tuple(step_metrics),
        final_state=state,
    )


def compute_simulation_summary(result: SimulationResult) -> SimulationSummary:
    """Compute aggregate metrics for a completed simulation run."""

    step_metrics = result.step_metrics
    completed_jobs = result.final_state.completed_jobs

    total_jobs_arrived = sum(step.jobs_arrived for step in step_metrics)
    total_jobs_completed = len(completed_jobs)

    if step_metrics:
        average_gpu_utilization = sum(
            step.gpu_utilization for step in step_metrics
        ) / len(step_metrics)

        average_queue_length = sum(
            step.jobs_waiting for step in step_metrics
        ) / len(step_metrics)

        average_scheduler_runtime_ms = sum(
            step.scheduler_runtime_ms for step in step_metrics
        ) / len(step_metrics)
    else:
        average_gpu_utilization = 0.0
        average_queue_length = 0.0
        average_scheduler_runtime_ms = 0.0

    total_priority_completed = sum(
        completed_job.job.priority for completed_job in completed_jobs
    )

    deadline_miss_count = sum(
        1
        for completed_job in completed_jobs
        if completed_job.completion_time > completed_job.job.deadline
    )

    deadline_miss_rate = (
        deadline_miss_count / total_jobs_completed
        if total_jobs_completed > 0
        else 0.0
    )

    total_wait_time = sum(
        completed_job.start_time - completed_job.job.release_time
        for completed_job in completed_jobs
    )

    average_completed_wait_time = (
        total_wait_time / total_jobs_completed
        if total_jobs_completed > 0
        else 0.0
    )

    return SimulationSummary(
        total_jobs_arrived=total_jobs_arrived,
        total_jobs_completed=total_jobs_completed,
        jobs_waiting_final=len(result.final_state.waiting_jobs),
        jobs_running_final=len(result.final_state.running_jobs),
        average_gpu_utilization=average_gpu_utilization,
        average_queue_length=average_queue_length,
        average_scheduler_runtime_ms=average_scheduler_runtime_ms,
        total_priority_completed=total_priority_completed,
        deadline_miss_count=deadline_miss_count,
        deadline_miss_rate=deadline_miss_rate,
        average_completed_wait_time=average_completed_wait_time,
    )


def _build_idle_decision() -> SchedulingDecision:
    return SchedulingDecision(
        assignments=(),
        started_job_ids=(),
        waiting_job_ids=(),
        objective_value=0.0,
        solver_status="idle",
        runtime_ms=0.0,
    )


def _build_step_metrics(
    time: int,
    jobs_arrived: int,
    decision: SchedulingDecision,
    decision_objective_value: float,
    total_priority_started: float,
    deadline_penalty_incurred: float,
    state_after_decision: SimulationState,
    jobs_completed: int,
    jobs_completed_this_step: int,
) -> SimulationStepMetrics:
    gpu_capacity_total = sum(node.total_gpus for node in state_after_decision.nodes)
    gpu_capacity_available = sum(
        node.available_gpus for node in state_after_decision.nodes
    )
    gpu_capacity_used = _running_gpu_count(state_after_decision.running_jobs)

    gpu_utilization = (
        gpu_capacity_used / gpu_capacity_total
        if gpu_capacity_total > 0
        else 0.0
    )

    return SimulationStepMetrics(
        time=time,
        jobs_arrived=jobs_arrived,
        jobs_started=len(decision.started_job_ids),
        jobs_waiting=len(state_after_decision.waiting_jobs),
        jobs_running=len(state_after_decision.running_jobs),
        jobs_completed=jobs_completed,
        jobs_completed_this_step=jobs_completed_this_step,
        gpu_capacity_total=gpu_capacity_total,
        gpu_capacity_available=gpu_capacity_available,
        gpu_capacity_used=gpu_capacity_used,
        gpu_utilization=gpu_utilization,
        total_priority_started=total_priority_started,
        deadline_penalty_incurred=deadline_penalty_incurred,
        objective_value=decision_objective_value,
        scheduler_status=decision.solver_status,
        scheduler_runtime_ms=decision.runtime_ms,
    )


def _running_gpu_count(running_jobs: tuple[RunningJob, ...]) -> int:
    return sum(
        assignment.gpu_count
        for running_job in running_jobs
        for assignment in running_job.assignments
    )
