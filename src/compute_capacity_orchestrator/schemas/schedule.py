"""Schemas for scheduler inputs and outputs."""

from __future__ import annotations
from dataclasses import dataclass

from compute_capacity_orchestrator.schemas.resources import ClusterNode
from compute_capacity_orchestrator.schemas.workload import JobRequest


@dataclass(frozen=True, slots=True)
class SchedulingSnapshot:
    """Current system state passed into a scheduling policy."""

    current_time: int
    queued_jobs: tuple[JobRequest, ...]
    nodes: tuple[ClusterNode, ...]

    def __post_init__(self) -> None:
        if self.current_time < 0:
            raise ValueError("SchedulingSnapshot current_time cannot be negative")

        if not self.nodes:
            raise ValueError("SchedulingSnapshot must contain at least one node")

        job_ids = [job.job_id for job in self.queued_jobs]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("SchedulingSnapshot job_id values must be unique")

        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("SchedulingSnapshot node_id values must be unique")

        for job in self.queued_jobs:
            if job.release_time > self.current_time:
                raise ValueError(
                    f"SchedulingSnapshot job {job.job_id}: "
                    "release_time cannot be later than current_time"
                )

@dataclass(frozen=True, slots=True)
class GpuAssignment:
    """GPU allocation from one node to one scheduled job."""

    job_id: str
    node_id: str
    gpu_count: int

    def __post_init__(self) -> None:
        job_ref = self.job_id or "<missing-job-id>"
        node_ref = self.node_id or "<missing-node-id>"

        if not self.job_id:
            raise ValueError("GpuAssignment job_id must be non-empty")

        if not self.node_id:
            raise ValueError(f"GpuAssignment {job_ref}: node_id must be non-empty")

        if self.gpu_count <= 0:
            raise ValueError(
                f"GpuAssignment job={job_ref}, node={node_ref}: "
                "gpu_count must be positive"
            )

@dataclass(frozen=True, slots=True)
class SchedulingDecision:
    """Output returned by a scheduling policy."""

    assignments: tuple[GpuAssignment, ...]
    started_job_ids: tuple[str, ...]
    waiting_job_ids: tuple[str, ...]
    objective_value: float
    solver_status: str
    runtime_ms: float

    def __post_init__(self) -> None:
        if self.runtime_ms < 0:
            raise ValueError("SchedulingDecision runtime_ms cannot be negative")

        if not self.solver_status:
            raise ValueError("SchedulingDecision solver_status must be non-empty")

        if len(self.started_job_ids) != len(set(self.started_job_ids)):
            raise ValueError("SchedulingDecision started_job_ids must be unique")

        if len(self.waiting_job_ids) != len(set(self.waiting_job_ids)):
            raise ValueError("SchedulingDecision waiting_job_ids must be unique")

        started = set(self.started_job_ids)
        waiting = set(self.waiting_job_ids)

        overlap = started & waiting
        if overlap:
            raise ValueError(
                f"SchedulingDecision job_ids cannot be both started and waiting: "
                f"{sorted(overlap)}"
            )

        assignment_pairs = [(a.job_id, a.node_id) for a in self.assignments]
        if len(assignment_pairs) != len(set(assignment_pairs)):
            raise ValueError(
                "SchedulingDecision assignments cannot contain duplicate "
                "(job_id, node_id) pairs"
            )

        assigned_jobs = {assignment.job_id for assignment in self.assignments}

        if assigned_jobs - started:
            raise ValueError(
                "SchedulingDecision assignments reference jobs that were not started"
            )

        if started - assigned_jobs:
            raise ValueError(
                "SchedulingDecision started jobs must have at least one assignment"
            )