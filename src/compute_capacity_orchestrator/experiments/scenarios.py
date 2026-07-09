"""Reproducible scheduling scenarios."""

from __future__ import annotations
from random import Random
from compute_capacity_orchestrator.schemas.resources import ClusterNode
from compute_capacity_orchestrator.schemas.schedule import SchedulingSnapshot
from compute_capacity_orchestrator.schemas.workload import JobRequest

def build_empty_capacity_snapshot() -> SchedulingSnapshot:
    """Build a snapshot with queued jobs but no available GPU capacity.

    Scenario summary:
        2 nodes, 2 queued jobs, 0 available GPUs.

    Key behavior:
        The cluster has installed GPUs, but no GPUs are currently available.
        A valid scheduler decision should start no jobs and leave all jobs waiting.
    """

    jobs = (
        JobRequest(
            job_id="urgent-small",
            gpu_demand=1,
            duration=3,
            priority=10.0,
            release_time=0,
            deadline=5,
        ),
        JobRequest(
            job_id="batch-medium",
            gpu_demand=4,
            duration=8,
            priority=16.0,
            release_time=0,
            deadline=20,
        ),
    )

    nodes = (
        ClusterNode(
            node_id="node-a",
            total_gpus=8,
            available_gpus=0,
            topology_group="rack-1",
        ),
        ClusterNode(
            node_id="node-b",
            total_gpus=8,
            available_gpus=0,
            topology_group="rack-1",
        ),
    )

    return SchedulingSnapshot(
        current_time=0,
        queued_jobs=jobs,
        nodes=nodes,
    )
def build_split_node_snapshot() -> SchedulingSnapshot:
    """Build a tiny snapshot where one job must split across two nodes.

    Scenario summary:
        2 nodes, 1 queued job, 2 total available GPUs.

    Key behavior:
        The job requires 2 GPUs.
        Each node has only 1 available GPU.
        Draft 0 placement allows the job to split across nodes.
    """

    jobs = (
        JobRequest(
            job_id="multi-node-job",
            gpu_demand=2,
            duration=5,
            priority=10.0,
            release_time=0,
            deadline=20,
        ),
    )

    nodes = (
        ClusterNode(
            node_id="node-a",
            total_gpus=8,
            available_gpus=1,
        ),
        ClusterNode(
            node_id="node-b",
            total_gpus=8,
            available_gpus=1,
        ),
    )

    return SchedulingSnapshot(
        current_time=0,
        queued_jobs=jobs,
        nodes=nodes,
    )

def build_small_snapshot() -> SchedulingSnapshot:
    """Build the default small snapshot used by demos and the dashboard."""

    return build_greedy_vs_mip_snapshot()

def build_value_density_snapshot() -> SchedulingSnapshot:
    """Build a capacity-limited snapshot where value density determines the order.

    Scenario summary:
        1 node, 3 queued jobs, 2 available GPUs.

    Key behavior:
        The large job has the highest raw priority but lower value density.
        Greedy value-density scheduling should start the two smaller dense jobs first.
    """

    jobs = (
        JobRequest(
            job_id="large-high-priority",
            gpu_demand=4,
            duration=10,
            priority=20.0,
            release_time=0,
            deadline=20,
        ),
        JobRequest(
            job_id="small-dense",
            gpu_demand=1,
            duration=5,
            priority=10.0,
            release_time=0,
            deadline=20,
        ),
        JobRequest(
            job_id="medium-dense",
            gpu_demand=1,
            duration=5,
            priority=8.0,
            release_time=0,
            deadline=20,
        ),
    )

    nodes = (
        ClusterNode(
            node_id="node-001",
            total_gpus=4,
            available_gpus=2,
        ),
    )

    return SchedulingSnapshot(
        current_time=0,
        queued_jobs=jobs,
        nodes=nodes,
    )

def build_capacity_limited_snapshot() -> SchedulingSnapshot:
    """Build a snapshot where greedy must skip an oversized job and continue.

    Scenario summary:
        1 node, 3 queued jobs, 6 available GPUs.

    Key behavior:
        Greedy starts the highest-density job first.
        The next job by tie-breaking is too large to fit.
        Greedy should leave it waiting and still start the remaining feasible job.
    """

    jobs = (
        JobRequest(
            job_id="job-high-density",
            gpu_demand=2,
            duration=5,
            priority=10.0,
            release_time=0,
            deadline=20,
        ),
        JobRequest(
            job_id="job-large",
            gpu_demand=7,
            duration=5,
            priority=14.0,
            release_time=0,
            deadline=20,
        ),
        JobRequest(
            job_id="job-medium",
            gpu_demand=4,
            duration=5,
            priority=8.0,
            release_time=0,
            deadline=20,
        ),
    )

    nodes = (
        ClusterNode(
            node_id="node-001",
            total_gpus=8,
            available_gpus=6,
        ),
    )

    return SchedulingSnapshot(
        current_time=0,
        queued_jobs=jobs,
        nodes=nodes,
    )

def build_greedy_vs_mip_snapshot() -> SchedulingSnapshot:
    """Build a small snapshot where greedy and exact MIP choose different jobs.

    Scenario summary:
        2 nodes, 4 queued jobs, 8 available GPUs.

    Key behavior:
        Value-density greedy starts the denser small jobs first.
        Exact MIP can start train-small and train-large together, using all 8 GPUs.
    """

    jobs = (
        JobRequest(
            job_id="train-small",
            gpu_demand=2,
            duration=6,
            priority=18.0,
            release_time=0,
            deadline=12,
        ),
        JobRequest(
            job_id="eval-batch",
            gpu_demand=1,
            duration=3,
            priority=9.0,
            release_time=0,
            deadline=2,
        ),
        JobRequest(
            job_id="train-large",
            gpu_demand=6,
            duration=10,
            priority=30.0,
            release_time=0,
            deadline=20,
        ),
        JobRequest(
            job_id="low-priority-sweep",
            gpu_demand=4,
            duration=8,
            priority=8.0,
            release_time=0,
            deadline=30,
        ),
    )

    nodes = (
        ClusterNode(
            node_id="node-a",
            total_gpus=8,
            available_gpus=5,
            topology_group="rack-1",
        ),
        ClusterNode(
            node_id="node-b",
            total_gpus=8,
            available_gpus=3,
            topology_group="rack-1",
        ),
    )

    return SchedulingSnapshot(
        current_time=0,
        queued_jobs=jobs,
        nodes=nodes,
    )

def build_snapshot_with_capacity(
    node_a_available: int,
    node_b_available: int,
) -> SchedulingSnapshot:
    """Build the small scenario with user-controlled available capacity."""

    base_snapshot = build_small_snapshot()

    nodes = (
        ClusterNode(
            node_id="node-a",
            total_gpus=8,
            available_gpus=node_a_available,
            topology_group="rack-1",
        ),
        ClusterNode(
            node_id="node-b",
            total_gpus=8,
            available_gpus=node_b_available,
            topology_group="rack-1",
        ),
    )

    return SchedulingSnapshot(
        current_time=base_snapshot.current_time,
        queued_jobs=base_snapshot.queued_jobs,
        nodes=nodes,
    )

def build_deadline_pressure_snapshot() -> SchedulingSnapshot:
    """Build a small snapshot where waiting deadline penalty changes the decision.

    Scenario summary:
        1 node, 1 available GPU, 2 queued jobs.

    Key behavior:
        The high-priority normal job has larger raw priority.
        The deadline-risk job has lower priority but creates a larger penalty if left waiting.
        Exact MIP should start deadline-risk when deadline_penalty_weight is high.
    """

    jobs = (
        JobRequest(
            job_id="high-priority-normal",
            gpu_demand=1,
            duration=1,
            priority=15.0,
            release_time=0,
            deadline=10,
        ),
        JobRequest(
            job_id="deadline-risk",
            gpu_demand=1,
            duration=5,
            priority=6.0,
            release_time=0,
            deadline=3,
        ),
    )

    nodes = (
        ClusterNode(
            node_id="node-001",
            total_gpus=1,
            available_gpus=1,
        ),
    )

    return SchedulingSnapshot(
        current_time=0,
        queued_jobs=jobs,
        nodes=nodes,
    )

def build_started_and_waiting_lateness_snapshot() -> SchedulingSnapshot:
    """Build a tiny snapshot for started-vs-waiting lateness accounting.

    Scenario summary:
        1 node, 2 queued jobs, 4 available GPUs.

    Key behavior:
        One job is intended to start and finish late.
        One job is intended to wait and become deadline-risky after one decision step.
        This scenario is used to test decision-level metrics, not scheduler choice.
    """

    jobs = (
        JobRequest(
            job_id="started-late",
            gpu_demand=2,
            duration=5,
            priority=20.0,
            release_time=0,
            deadline=3,
        ),
        JobRequest(
            job_id="waiting-risky",
            gpu_demand=2,
            duration=3,
            priority=9.0,
            release_time=0,
            deadline=2,
        ),
    )

    nodes = (
        ClusterNode(
            node_id="node-001",
            total_gpus=4,
            available_gpus=4,
        ),
    )

    return SchedulingSnapshot(
        current_time=0,
        queued_jobs=jobs,
        nodes=nodes,
    )



def build_random_snapshot(
    num_nodes: int,
    num_jobs: int,
    seed: int,
    gpu_demand_range: tuple[int, int] = (1, 8),
    duration_range: tuple[int, int] = (1, 24),
    priority_range: tuple[float, float] = (1.0, 100.0),
    deadline_slack_range: tuple[int, int] = (-4, 24),
    total_gpus_per_node: int = 8,
    available_capacity_ratio: float = 1.0,
) -> SchedulingSnapshot:
    """Build a reproducible random snapshot for scale and stress experiments.

    Scenario summary:
        num_nodes compute nodes and num_jobs queued jobs.

    Key behavior:
        Uses a deterministic random seed.
        All jobs are released at current_time=0.
        available_capacity_ratio controls currently available GPUs per node.
    """

    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    if num_jobs < 0:
        raise ValueError("num_jobs cannot be negative")

    if total_gpus_per_node <= 0:
        raise ValueError("total_gpus_per_node must be positive")

    if not 0.0 <= available_capacity_ratio <= 1.0:
        raise ValueError("available_capacity_ratio must be between 0.0 and 1.0")

    _validate_int_range("gpu_demand_range", gpu_demand_range, minimum=1)
    _validate_int_range("duration_range", duration_range, minimum=1)
    _validate_float_range("priority_range", priority_range, minimum=0.0)
    _validate_int_range("deadline_slack_range", deadline_slack_range, minimum=None)

    rng = Random(seed)
    current_time = 0

    available_gpus_per_node = int(round(total_gpus_per_node * available_capacity_ratio))

    nodes = tuple(
        ClusterNode(
            node_id=f"node-{node_index:05d}",
            total_gpus=total_gpus_per_node,
            available_gpus=available_gpus_per_node,
            topology_group=f"rack-{node_index % 4}",
        )
        for node_index in range(num_nodes)
    )

    jobs = []

    for job_index in range(num_jobs):
        #gpu_demand = rng.randint(gpu_demand_range[0], gpu_demand_range[1])
        pareto_shape = 1.1
        raw_demand = gpu_demand_range[0] * rng.paretovariate(pareto_shape)
        gpu_demand = int(round(raw_demand))
        gpu_demand = max(gpu_demand_range[0], gpu_demand)
        gpu_demand = min(gpu_demand_range[1], gpu_demand)
        duration = rng.randint(duration_range[0], duration_range[1])
        priority = rng.uniform(priority_range[0], priority_range[1])
        deadline_slack = rng.randint(
            deadline_slack_range[0],
            deadline_slack_range[1],
        )

        deadline = max(
            current_time,
            current_time + duration + deadline_slack,
        )

        jobs.append(
            JobRequest(
                job_id=f"job-{job_index:05d}",
                gpu_demand=gpu_demand,
                duration=duration,
                priority=round(priority, 3),
                release_time=current_time,
                deadline=deadline,
            )
        )

    return SchedulingSnapshot(
        current_time=current_time,
        queued_jobs=tuple(jobs),
        nodes=nodes,
    )


def _validate_int_range(
    name: str,
    value_range: tuple[int, int],
    minimum: int | None,
) -> None:
    lower, upper = value_range

    if lower > upper:
        raise ValueError(f"{name} lower bound cannot exceed upper bound")

    if minimum is not None and lower < minimum:
        raise ValueError(f"{name} lower bound must be at least {minimum}")


def _validate_float_range(
    name: str,
    value_range: tuple[float, float],
    minimum: float | None,
) -> None:
    lower, upper = value_range

    if lower > upper:
        raise ValueError(f"{name} lower bound cannot exceed upper bound")

    if minimum is not None and lower < minimum:
        raise ValueError(f"{name} lower bound must be at least {minimum}")