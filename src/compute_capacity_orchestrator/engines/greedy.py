"""Greedy scheduling policies."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from time import perf_counter

from compute_capacity_orchestrator.schemas.schedule import (
    GpuAssignment,
    SchedulingDecision,
    SchedulingSnapshot,
)
from compute_capacity_orchestrator.schemas.workload import JobRequest


@dataclass(frozen=True, slots=True)
class GreedyScheduler:
    """Value-density greedy scheduler for a single scheduling snapshot."""

    def solve(self, snapshot: SchedulingSnapshot) -> SchedulingDecision:
        start_time = perf_counter()

        remaining_by_node = {
            node.node_id: node.available_gpus for node in snapshot.nodes
        }
        total_remaining_gpus = sum(remaining_by_node.values())
        heap: list[tuple[float, float, int, int, JobRequest]] = []

        for index, job in enumerate(snapshot.queued_jobs):
            value_density = job.priority / job.gpu_demand

            heapq.heappush(
                heap,
                (-value_density, -job.priority, job.gpu_demand, index, job),
            )

        assignments: list[GpuAssignment] = []
        started_job_ids: list[str] = []
        waiting_job_ids: list[str] = []
        objective_value = 0.0

        while heap:
            _, _, _, _, job = heapq.heappop(heap)

            if total_remaining_gpus < job.gpu_demand:
                waiting_job_ids.append(job.job_id)
                continue

            job_assignments = self._allocate_job(job, remaining_by_node)

            assignments.extend(job_assignments)
            started_job_ids.append(job.job_id)
            objective_value += job.priority
            total_remaining_gpus -= job.gpu_demand

        runtime_ms = (perf_counter() - start_time) * 1000.0

        return SchedulingDecision(
            assignments=tuple(assignments),
            started_job_ids=tuple(started_job_ids),
            waiting_job_ids=tuple(waiting_job_ids),
            objective_value=objective_value,
            solver_status="greedy_value_density",
            runtime_ms=runtime_ms,
        )

    def _allocate_job(
        self,
        job: JobRequest,
        remaining_by_node: dict[str, int],
    ) -> list[GpuAssignment]:
        needed = job.gpu_demand
        assignments: list[GpuAssignment] = []

        for node_id, available in remaining_by_node.items():
            if needed == 0:
                break

            gpu_count = min(available, needed)

            if gpu_count == 0:
                continue

            assignments.append(
                GpuAssignment(
                    job_id=job.job_id,
                    node_id=node_id,
                    gpu_count=gpu_count,
                )
            )

            remaining_by_node[node_id] -= gpu_count
            needed -= gpu_count

        if needed != 0:
            raise RuntimeError(
                f"GreedyScheduler failed to fully allocate job {job.job_id}"
            )

        return assignments