"""State objects for scheduling simulation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from compute_capacity_orchestrator.schemas.resources import ClusterNode
from compute_capacity_orchestrator.schemas.schedule import GpuAssignment
from compute_capacity_orchestrator.schemas.workload import JobRequest


@dataclass(frozen=True, slots=True)
class RunningJob:
    """A job that has started and is consuming GPU capacity."""

    job: JobRequest
    assignments: tuple[GpuAssignment, ...]
    start_time: int
    completion_time: int

    def __post_init__(self) -> None:
        if self.start_time < 0:
            raise ValueError("RunningJob start_time cannot be negative")

        if self.start_time < self.job.release_time:
            raise ValueError(
                f"RunningJob {self.job.job_id}: start_time cannot be earlier "
                "than job release_time"
            )

        expected_completion_time = self.start_time + self.job.duration
        if self.completion_time != expected_completion_time:
            raise ValueError(
                f"RunningJob {self.job.job_id}: completion_time must equal "
                "start_time + duration"
            )

        if not self.assignments:
            raise ValueError(
                f"RunningJob {self.job.job_id}: assignments cannot be empty"
            )

        assigned_gpus = 0

        for assignment in self.assignments:
            if assignment.job_id != self.job.job_id:
                raise ValueError(
                    f"RunningJob {self.job.job_id}: assignment references "
                    f"different job_id {assignment.job_id}"
                )

            assigned_gpus += assignment.gpu_count

        if assigned_gpus != self.job.gpu_demand:
            raise ValueError(
                f"RunningJob {self.job.job_id}: assigned_gpus={assigned_gpus}, "
                f"gpu_demand={self.job.gpu_demand}"
            )


@dataclass(frozen=True, slots=True)
class SimulationState:
    """Complete state of the simulated scheduling system at one time."""

    current_time: int
    waiting_jobs: tuple[JobRequest, ...]
    running_jobs: tuple[RunningJob, ...]
    nodes: tuple[ClusterNode, ...]
    completed_jobs: tuple[RunningJob, ...] = ()

    def __post_init__(self) -> None:
        if self.current_time < 0:
            raise ValueError("SimulationState current_time cannot be negative")

        if not self.nodes:
            raise ValueError("SimulationState must contain at least one node")

        waiting_job_ids = [job.job_id for job in self.waiting_jobs]
        running_job_ids = [running_job.job.job_id for running_job in self.running_jobs]
        completed_job_ids = [
            completed_job.job.job_id for completed_job in self.completed_jobs
        ]
        node_ids = [node.node_id for node in self.nodes]

        _validate_unique_ids("waiting job_ids", waiting_job_ids)
        _validate_unique_ids("running job_ids", running_job_ids)
        _validate_unique_ids("completed job_ids", completed_job_ids)
        _validate_unique_ids("node_ids", node_ids)

        waiting_set = set(waiting_job_ids)
        running_set = set(running_job_ids)
        completed_set = set(completed_job_ids)

        if waiting_set & running_set:
            raise ValueError(
                "SimulationState jobs cannot be both waiting and running: "
                f"{sorted(waiting_set & running_set)}"
            )

        if waiting_set & completed_set:
            raise ValueError(
                "SimulationState jobs cannot be both waiting and completed: "
                f"{sorted(waiting_set & completed_set)}"
            )

        if running_set & completed_set:
            raise ValueError(
                "SimulationState jobs cannot be both running and completed: "
                f"{sorted(running_set & completed_set)}"
            )

        for job in self.waiting_jobs:
            if job.release_time > self.current_time:
                raise ValueError(
                    f"SimulationState waiting job {job.job_id}: release_time "
                    "cannot be later than current_time"
                )

        nodes_by_id = {node.node_id: node for node in self.nodes}
        assigned_gpus_by_node: dict[str, int] = defaultdict(int)

        for running_job in self.running_jobs:
            if running_job.start_time > self.current_time:
                raise ValueError(
                    f"SimulationState running job {running_job.job.job_id}: "
                    "start_time cannot be later than current_time"
                )

            if running_job.completion_time <= self.current_time:
                raise ValueError(
                    f"SimulationState running job {running_job.job.job_id}: "
                    "completion_time must be later than current_time"
                )

            for assignment in running_job.assignments:
                if assignment.node_id not in nodes_by_id:
                    raise ValueError(
                        f"SimulationState running job {running_job.job.job_id}: "
                        f"assignment references unknown node_id {assignment.node_id}"
                    )

                assigned_gpus_by_node[assignment.node_id] += assignment.gpu_count

        for completed_job in self.completed_jobs:
            if completed_job.completion_time > self.current_time:
                raise ValueError(
                    f"SimulationState completed job {completed_job.job.job_id}: "
                    "completion_time cannot be later than current_time"
                )

        for node_id, assigned_gpus in assigned_gpus_by_node.items():
            node = nodes_by_id[node_id]

            if assigned_gpus + node.available_gpus > node.total_gpus:
                raise ValueError(
                    f"SimulationState node {node_id}: running assignments plus "
                    "available_gpus cannot exceed total_gpus"
                )


def _validate_unique_ids(name: str, ids: list[str]) -> None:
    if len(ids) != len(set(ids)):
        raise ValueError(f"SimulationState {name} must be unique")