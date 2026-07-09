"""Schemas for workload requests."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class JobRequest:
    """A workload request waiting for GPU capacity."""

    job_id: str
    gpu_demand: int
    duration: int
    priority: float
    release_time: int
    deadline: int

    def __post_init__(self) -> None:
        job_ref = self.job_id or "<missing-job-id>"

        if not self.job_id:
            raise ValueError("JobRequest job_id must be non-empty")

        if self.gpu_demand <= 0:
            raise ValueError(f"JobRequest {job_ref}: gpu_demand must be positive")

        if self.duration <= 0:
            raise ValueError(f"JobRequest {job_ref}: duration must be positive")

        if self.priority < 0:
            raise ValueError(f"JobRequest {job_ref}: priority cannot be negative")

        if self.release_time < 0:
            raise ValueError(f"JobRequest {job_ref}: release_time cannot be negative")

        if self.deadline < 0:
            raise ValueError(f"JobRequest {job_ref}: deadline cannot be negative")

        if self.deadline < self.release_time:
            raise ValueError(
                f"JobRequest {job_ref}: deadline cannot be earlier than release_time"
            )