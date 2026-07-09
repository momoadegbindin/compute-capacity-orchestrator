"""Metrics for validated scheduling decisions."""

from __future__ import annotations

from dataclasses import dataclass

from compute_capacity_orchestrator.schemas.schedule import (
    SchedulingDecision,
    SchedulingSnapshot,
)


@dataclass(frozen=True, slots=True)
class DecisionMetrics:
    """Summary metrics for one validated scheduling decision."""

    jobs_submitted: int
    jobs_started: int
    jobs_waiting: int
    gpu_capacity_available: int
    gpu_capacity_used: int
    gpu_utilization: float
    total_priority_started: float
    started_deadline_penalty: float
    waiting_deadline_penalty: float
    deadline_penalty_incurred: float
    objective_value: float
    scheduler_objective_value: float
    solver_status: str
    runtime_ms: float


def compute_decision_metrics(
    snapshot: SchedulingSnapshot,
    decision: SchedulingDecision,
    deadline_penalty_weight: float = 0.0,
    decision_step: int = 1,
) -> DecisionMetrics:
    """Compute metrics for a validated scheduling decision."""

    if deadline_penalty_weight < 0:
        raise ValueError("deadline_penalty_weight cannot be negative")

    if decision_step <= 0:
        raise ValueError("decision_step must be positive")

    jobs_by_id = {job.job_id: job for job in snapshot.queued_jobs}

    gpu_capacity_available = sum(node.available_gpus for node in snapshot.nodes)
    gpu_capacity_used = sum(assignment.gpu_count for assignment in decision.assignments)

    gpu_utilization = (
        gpu_capacity_used / gpu_capacity_available
        if gpu_capacity_available > 0
        else 0.0
    )

    total_priority_started = sum(
        jobs_by_id[job_id].priority for job_id in decision.started_job_ids
    )

    started_deadline_penalty = 0.0
    for job_id in decision.started_job_ids:
        job = jobs_by_id[job_id]
        lateness = max(0, snapshot.current_time + job.duration - job.deadline)
        started_deadline_penalty += deadline_penalty_weight * lateness

    waiting_deadline_penalty = 0.0
    for job_id in decision.waiting_job_ids:
        job = jobs_by_id[job_id]
        lateness = max(
            0,
            snapshot.current_time + decision_step + job.duration - job.deadline,
        )
        waiting_deadline_penalty += deadline_penalty_weight * lateness

    deadline_penalty_incurred = (
        started_deadline_penalty + waiting_deadline_penalty
    )

    objective_value = total_priority_started - deadline_penalty_incurred

    return DecisionMetrics(
        jobs_submitted=len(snapshot.queued_jobs),
        jobs_started=len(decision.started_job_ids),
        jobs_waiting=len(decision.waiting_job_ids),
        gpu_capacity_available=gpu_capacity_available,
        gpu_capacity_used=gpu_capacity_used,
        gpu_utilization=gpu_utilization,
        total_priority_started=total_priority_started,
        started_deadline_penalty=started_deadline_penalty,
        waiting_deadline_penalty=waiting_deadline_penalty,
        deadline_penalty_incurred=deadline_penalty_incurred,
        objective_value=objective_value,
        scheduler_objective_value=decision.objective_value,
        solver_status=decision.solver_status,
        runtime_ms=decision.runtime_ms,
    )