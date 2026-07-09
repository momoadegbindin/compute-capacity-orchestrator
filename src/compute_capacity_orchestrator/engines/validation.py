"""Validation for scheduler decisions."""

from __future__ import annotations

from collections import defaultdict

from compute_capacity_orchestrator.schemas.schedule import (
    SchedulingDecision,
    SchedulingSnapshot,
)


def validate_decision(
    snapshot: SchedulingSnapshot,
    decision: SchedulingDecision,
) -> None:
    """Validate that a scheduling decision is feasible for a snapshot."""

    jobs_by_id = {job.job_id: job for job in snapshot.queued_jobs}
    nodes_by_id = {node.node_id: node for node in snapshot.nodes}

    snapshot_job_ids = set(jobs_by_id)
    decision_job_ids = set(decision.started_job_ids) | set(decision.waiting_job_ids)

    unknown_jobs = decision_job_ids - snapshot_job_ids
    if unknown_jobs:
        raise ValueError(
            f"SchedulingDecision references unknown job_ids: {sorted(unknown_jobs)}"
        )

    missing_jobs = snapshot_job_ids - decision_job_ids
    if missing_jobs:
        raise ValueError(
            f"SchedulingDecision is missing queued job_ids: {sorted(missing_jobs)}"
        )

    started_job_ids = set(decision.started_job_ids)

    assigned_gpus_by_job: dict[str, int] = defaultdict(int)
    assigned_gpus_by_node: dict[str, int] = defaultdict(int)

    for assignment in decision.assignments:
        if assignment.job_id not in jobs_by_id:
            raise ValueError(
                f"GpuAssignment references unknown job_id: {assignment.job_id}"
            )

        if assignment.node_id not in nodes_by_id:
            raise ValueError(
                f"GpuAssignment references unknown node_id: {assignment.node_id}"
            )

        if assignment.job_id not in started_job_ids:
            raise ValueError(
                f"GpuAssignment references job {assignment.job_id}, "
                "but that job was not started"
            )

        assigned_gpus_by_job[assignment.job_id] += assignment.gpu_count
        assigned_gpus_by_node[assignment.node_id] += assignment.gpu_count

    for job_id in decision.started_job_ids:
        assigned = assigned_gpus_by_job[job_id]
        required = jobs_by_id[job_id].gpu_demand

        if assigned != required:
            raise ValueError(
                f"SchedulingDecision job {job_id}: assigned_gpus={assigned}, "
                f"gpu_demand={required}"
            )

    for node_id, assigned in assigned_gpus_by_node.items():
        available = nodes_by_id[node_id].available_gpus

        if assigned > available:
            raise ValueError(
                f"SchedulingDecision node {node_id}: assigned_gpus={assigned}, "
                f"available_gpus={available}"
            )