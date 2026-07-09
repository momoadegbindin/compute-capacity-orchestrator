"""State transitions for scheduling simulation."""

from __future__ import annotations

from collections import defaultdict

from compute_capacity_orchestrator.engines.validation import validate_decision
from compute_capacity_orchestrator.schemas.resources import ClusterNode
from compute_capacity_orchestrator.schemas.schedule import (
    GpuAssignment,
    SchedulingDecision,
    SchedulingSnapshot,
)
from compute_capacity_orchestrator.schemas.workload import JobRequest
from compute_capacity_orchestrator.simulation.state import RunningJob, SimulationState


def build_scheduling_snapshot(state: SimulationState) -> SchedulingSnapshot:
    """Build the scheduler-facing snapshot from the full simulation state."""

    return SchedulingSnapshot(
        current_time=state.current_time,
        queued_jobs=state.waiting_jobs,
        nodes=state.nodes,
    )


def apply_scheduling_decision(
    state: SimulationState,
    decision: SchedulingDecision,
) -> SimulationState:
    """Apply a validated scheduling decision to the simulation state."""

    snapshot = build_scheduling_snapshot(state)
    validate_decision(snapshot, decision)

    jobs_by_id = {job.job_id: job for job in state.waiting_jobs}
    waiting_job_ids = set(decision.waiting_job_ids)

    assignments_by_job: dict[str, list[GpuAssignment]] = defaultdict(list)
    assigned_gpus_by_node: dict[str, int] = defaultdict(int)

    for assignment in decision.assignments:
        assignments_by_job[assignment.job_id].append(assignment)
        assigned_gpus_by_node[assignment.node_id] += assignment.gpu_count

    newly_running_jobs = tuple(
        RunningJob(
            job=jobs_by_id[job_id],
            assignments=tuple(assignments_by_job[job_id]),
            start_time=state.current_time,
            completion_time=state.current_time + jobs_by_id[job_id].duration,
        )
        for job_id in decision.started_job_ids
    )

    updated_nodes = tuple(
        _with_available_gpus(
            node=node,
            available_gpus=node.available_gpus - assigned_gpus_by_node[node.node_id],
        )
        for node in state.nodes
    )

    updated_waiting_jobs = tuple(
        job for job in state.waiting_jobs if job.job_id in waiting_job_ids
    )

    return SimulationState(
        current_time=state.current_time,
        waiting_jobs=updated_waiting_jobs,
        running_jobs=state.running_jobs + newly_running_jobs,
        nodes=updated_nodes,
        completed_jobs=state.completed_jobs,
    )


def advance_time(
    state: SimulationState,
    next_time: int,
) -> SimulationState:
    """Advance simulation time and release capacity from completed jobs."""

    if next_time <= state.current_time:
        raise ValueError("next_time must be later than current_time")

    remaining_running_jobs: list[RunningJob] = []
    newly_completed_jobs: list[RunningJob] = []
    released_gpus_by_node: dict[str, int] = defaultdict(int)

    for running_job in state.running_jobs:
        if running_job.completion_time <= next_time:
            newly_completed_jobs.append(running_job)

            for assignment in running_job.assignments:
                released_gpus_by_node[assignment.node_id] += assignment.gpu_count
        else:
            remaining_running_jobs.append(running_job)

    updated_nodes = tuple(
        _with_available_gpus(
            node=node,
            available_gpus=node.available_gpus + released_gpus_by_node[node.node_id],
        )
        for node in state.nodes
    )

    return SimulationState(
        current_time=next_time,
        waiting_jobs=state.waiting_jobs,
        running_jobs=tuple(remaining_running_jobs),
        nodes=updated_nodes,
        completed_jobs=state.completed_jobs + tuple(newly_completed_jobs),
    )


def add_arrivals(
    state: SimulationState,
    arriving_jobs: tuple[JobRequest, ...],
) -> SimulationState:
    """Add newly released jobs to the waiting queue."""

    for job in arriving_jobs:
        if job.release_time > state.current_time:
            raise ValueError(
                f"Arriving job {job.job_id}: release_time cannot be later "
                "than current_time"
            )

    return SimulationState(
        current_time=state.current_time,
        waiting_jobs=state.waiting_jobs + arriving_jobs,
        running_jobs=state.running_jobs,
        nodes=state.nodes,
        completed_jobs=state.completed_jobs,
    )


def _with_available_gpus(
    node: ClusterNode,
    available_gpus: int,
) -> ClusterNode:
    return ClusterNode(
        node_id=node.node_id,
        total_gpus=node.total_gpus,
        available_gpus=available_gpus,
        topology_group=node.topology_group,
    )
