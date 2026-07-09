from __future__ import annotations

import pytest

from compute_capacity_orchestrator.simulation.arrivals import (
    generate_heavy_tailed_arrivals_by_time,
)


def test_generate_heavy_tailed_arrivals_is_reproducible() -> None:
    arrivals_a = generate_heavy_tailed_arrivals_by_time(
        start_time=0,
        horizon=10,
        seed=7,
        arrival_rate=2.0,
    )

    arrivals_b = generate_heavy_tailed_arrivals_by_time(
        start_time=0,
        horizon=10,
        seed=7,
        arrival_rate=2.0,
    )

    assert arrivals_a == arrivals_b


def test_generate_heavy_tailed_arrivals_respects_bounds() -> None:
    arrivals = generate_heavy_tailed_arrivals_by_time(
        start_time=5,
        horizon=10,
        seed=11,
        arrival_rate=3.0,
        gpu_demand_range=(1, 32),
        duration_range=(2, 12),
        priority_range=(5.0, 25.0),
        deadline_slack_range=(-2, 8),
    )

    all_jobs = [
        job
        for jobs_at_time in arrivals.values()
        for job in jobs_at_time
    ]

    assert all_jobs

    for time, jobs_at_time in arrivals.items():
        assert 5 <= time < 15

        for job in jobs_at_time:
            assert job.release_time == time
            assert 1 <= job.gpu_demand <= 32
            assert 2 <= job.duration <= 12
            assert 5.0 <= job.priority <= 25.0
            assert job.deadline >= job.release_time


def test_generate_heavy_tailed_arrivals_uses_unique_job_ids() -> None:
    arrivals = generate_heavy_tailed_arrivals_by_time(
        start_time=0,
        horizon=20,
        seed=13,
        arrival_rate=4.0,
    )

    job_ids = [
        job.job_id
        for jobs_at_time in arrivals.values()
        for job in jobs_at_time
    ]

    assert len(job_ids) == len(set(job_ids))


def test_generate_heavy_tailed_arrivals_allows_zero_arrival_rate() -> None:
    arrivals = generate_heavy_tailed_arrivals_by_time(
        start_time=0,
        horizon=10,
        seed=7,
        arrival_rate=0.0,
    )

    assert arrivals == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "start_time": -1,
            "horizon": 10,
            "seed": 1,
            "arrival_rate": 1.0,
        },
        {
            "start_time": 0,
            "horizon": 0,
            "seed": 1,
            "arrival_rate": 1.0,
        },
        {
            "start_time": 0,
            "horizon": 10,
            "seed": 1,
            "arrival_rate": -1.0,
        },
        {
            "start_time": 0,
            "horizon": 10,
            "seed": 1,
            "arrival_rate": 1.0,
            "gpu_demand_range": (0, 8),
        },
        {
            "start_time": 0,
            "horizon": 10,
            "seed": 1,
            "arrival_rate": 1.0,
            "duration_range": (0, 8),
        },
        {
            "start_time": 0,
            "horizon": 10,
            "seed": 1,
            "arrival_rate": 1.0,
            "priority_range": (-1.0, 10.0),
        },
        {
            "start_time": 0,
            "horizon": 10,
            "seed": 1,
            "arrival_rate": 1.0,
            "max_arrivals_per_step": 0,
        },
    ],
)
def test_generate_heavy_tailed_arrivals_rejects_invalid_parameters(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        generate_heavy_tailed_arrivals_by_time(**kwargs)