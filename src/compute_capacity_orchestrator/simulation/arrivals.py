"""Workload generation for scheduling simulation."""

from __future__ import annotations

import math
from random import Random

from compute_capacity_orchestrator.schemas.workload import JobRequest


PARETO_SHAPE = 1.5


def generate_heavy_tailed_arrivals_by_time(
    start_time: int,
    horizon: int,
    seed: int,
    arrival_rate: float,
    gpu_demand_range: tuple[int, int] = (1, 16),
    duration_range: tuple[int, int] = (1, 24),
    priority_range: tuple[float, float] = (1.0, 100.0),
    deadline_slack_range: tuple[int, int] = (-4, 24),
    max_arrivals_per_step: int = 20,
) -> dict[int, tuple[JobRequest, ...]]:
    """Generate reproducible stochastic job arrivals over a simulation horizon.

    Arrival counts are sampled from a Poisson distribution.
    GPU demand is heavy-tailed: many small jobs, rare large jobs.
    """

    if start_time < 0:
        raise ValueError("start_time cannot be negative")

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if arrival_rate < 0:
        raise ValueError("arrival_rate cannot be negative")

    if max_arrivals_per_step <= 0:
        raise ValueError("max_arrivals_per_step must be positive")

    _validate_int_range("gpu_demand_range", gpu_demand_range, minimum=1)
    _validate_int_range("duration_range", duration_range, minimum=1)
    _validate_float_range("priority_range", priority_range, minimum=0.0)
    _validate_int_range("deadline_slack_range", deadline_slack_range, minimum=None)

    rng = Random(seed)
    arrivals_by_time: dict[int, tuple[JobRequest, ...]] = {}
    next_job_index = 0

    for time in range(start_time, start_time + horizon):
        arrivals_count = _sample_poisson(rng, arrival_rate)
        arrivals_count = min(arrivals_count, max_arrivals_per_step)

        jobs: list[JobRequest] = []

        for _ in range(arrivals_count):
            gpu_demand = _sample_bounded_pareto_int(
                rng=rng,
                lower=gpu_demand_range[0],
                upper=gpu_demand_range[1],
            )
            duration = rng.randint(duration_range[0], duration_range[1])
            priority = round(
                rng.uniform(priority_range[0], priority_range[1]),
                3,
            )
            deadline_slack = rng.randint(
                deadline_slack_range[0],
                deadline_slack_range[1],
            )

            deadline = max(time, time + duration + deadline_slack)

            jobs.append(
                JobRequest(
                    job_id=f"job-{next_job_index:06d}",
                    gpu_demand=gpu_demand,
                    duration=duration,
                    priority=priority,
                    release_time=time,
                    deadline=deadline,
                )
            )

            next_job_index += 1

        if jobs:
            arrivals_by_time[time] = tuple(jobs)

    return arrivals_by_time


def _sample_poisson(rng: Random, rate: float) -> int:
    if rate == 0:
        return 0

    threshold = math.exp(-rate)
    product = 1.0
    count = 0

    while product > threshold:
        count += 1
        product *= rng.random()

    return count - 1


def _sample_bounded_pareto_int(
    rng: Random,
    lower: int,
    upper: int,
) -> int:
    raw_value = lower * rng.paretovariate(PARETO_SHAPE)
    value = int(round(raw_value))

    return max(lower, min(value, upper))


def _validate_int_range(
    name: str,
    value_range: tuple[int, int],
    minimum: int | None,
) -> None:
    lower, upper = value_range

    if minimum is not None and lower < minimum:
        raise ValueError(f"{name} lower bound must be at least {minimum}")

    if upper < lower:
        raise ValueError(f"{name} upper bound cannot be less than lower bound")


def _validate_float_range(
    name: str,
    value_range: tuple[float, float],
    minimum: float | None,
) -> None:
    lower, upper = value_range

    if minimum is not None and lower < minimum:
        raise ValueError(f"{name} lower bound must be at least {minimum}")

    if upper < lower:
        raise ValueError(f"{name} upper bound cannot be less than lower bound")