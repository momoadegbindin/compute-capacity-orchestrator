from __future__ import annotations
import pytest
from compute_capacity_orchestrator.schemas.workload import JobRequest


def test_job_request_validation() -> None:
    job = JobRequest(
        job_id="job-001",
        gpu_demand=4,
        duration=10,
        priority=25.0,
        release_time=0,
        deadline=20,
    )

    assert job.job_id == "job-001"
    assert job.gpu_demand == 4
    assert job.duration == 10
    assert job.priority == 25.0
    assert job.release_time == 0
    assert job.deadline == 20

    with pytest.raises(ValueError) as exc_info:
        JobRequest(
            job_id="job-002",
            gpu_demand=0,
            duration=10,
            priority=25.0,
            release_time=0,
            deadline=20,
        )

    assert "job-002" in str(exc_info.value)
    assert "gpu_demand" in str(exc_info.value)