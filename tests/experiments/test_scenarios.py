from __future__ import annotations

from compute_capacity_orchestrator.experiments.scenarios import (
    build_empty_capacity_snapshot,
    build_split_node_snapshot,
    build_greedy_vs_mip_snapshot,
    build_small_snapshot,
    build_deadline_pressure_snapshot,
    build_value_density_snapshot,
    build_capacity_limited_snapshot,
    build_started_and_waiting_lateness_snapshot,
    build_random_snapshot,
)

def test_build_empty_capacity_snapshot_has_expected_shape() -> None:
    # Empty-capacity snapshot:
    # 2 nodes, 2 queued jobs, installed GPUs exist but available GPUs are zero.
    snapshot = build_empty_capacity_snapshot()

    assert snapshot.current_time == 0
    assert len(snapshot.nodes) == 2
    assert len(snapshot.queued_jobs) == 2

    assert [node.node_id for node in snapshot.nodes] == ["node-a", "node-b"]
    assert [node.total_gpus for node in snapshot.nodes] == [8, 8]
    assert [node.available_gpus for node in snapshot.nodes] == [0, 0]
    assert sum(node.available_gpus for node in snapshot.nodes) == 0

    assert [job.job_id for job in snapshot.queued_jobs] == [
        "urgent-small",
        "batch-medium",
    ]

    jobs_by_id = {job.job_id: job for job in snapshot.queued_jobs}

    assert jobs_by_id["urgent-small"].gpu_demand == 1
    assert jobs_by_id["urgent-small"].priority == 10.0

    assert jobs_by_id["batch-medium"].gpu_demand == 4
    assert jobs_by_id["batch-medium"].priority == 16.0

def test_build_split_node_snapshot_has_expected_shape() -> None:
    # Split-node snapshot:
    # 2 nodes with 1 available GPU each; one job requires 2 GPUs.
    snapshot = build_split_node_snapshot()

    assert snapshot.current_time == 0
    assert len(snapshot.nodes) == 2
    assert len(snapshot.queued_jobs) == 1

    assert [node.node_id for node in snapshot.nodes] == ["node-a", "node-b"]
    assert [node.available_gpus for node in snapshot.nodes] == [1, 1]
    assert sum(node.available_gpus for node in snapshot.nodes) == 2

    job = snapshot.queued_jobs[0]

    assert job.job_id == "multi-node-job"
    assert job.gpu_demand == 2
    assert job.priority == 10.0
    assert job.deadline == 20

def test_build_greedy_vs_mip_snapshot_has_expected_shape() -> None:
    # Greedy-vs-MIP snapshot:
    # 2 nodes, 4 jobs, 8 available GPUs; exact MIP can pack train-small + train-large.
    snapshot = build_greedy_vs_mip_snapshot()

    assert snapshot.current_time == 0
    assert len(snapshot.nodes) == 2
    assert len(snapshot.queued_jobs) == 4

    assert [node.node_id for node in snapshot.nodes] == ["node-a", "node-b"]
    assert [node.available_gpus for node in snapshot.nodes] == [5, 3]
    assert sum(node.available_gpus for node in snapshot.nodes) == 8

    assert [job.job_id for job in snapshot.queued_jobs] == [
        "train-small",
        "eval-batch",
        "train-large",
        "low-priority-sweep",
    ]

    jobs_by_id = {job.job_id: job for job in snapshot.queued_jobs}

    assert jobs_by_id["train-small"].gpu_demand == 2
    assert jobs_by_id["train-small"].priority == 18.0

    assert jobs_by_id["train-large"].gpu_demand == 6
    assert jobs_by_id["train-large"].priority == 30.0

    assert jobs_by_id["eval-batch"].deadline == 2


def test_build_small_snapshot_preserves_default_demo_scenario() -> None:
    # Backward-compatibility check:
    # existing scripts and Streamlit still call build_small_snapshot().
    small_snapshot = build_small_snapshot()
    named_snapshot = build_greedy_vs_mip_snapshot()

    assert small_snapshot == named_snapshot

def test_build_deadline_pressure_snapshot_has_expected_shape() -> None:
    # Deadline-pressure snapshot:
    # 1 node, 1 available GPU, 2 jobs competing for capacity.
    # deadline-risk has lower priority but becomes expensive to leave waiting.
    snapshot = build_deadline_pressure_snapshot()

    assert snapshot.current_time == 0
    assert len(snapshot.nodes) == 1
    assert len(snapshot.queued_jobs) == 2

    assert snapshot.nodes[0].node_id == "node-001"
    assert snapshot.nodes[0].available_gpus == 1
    assert snapshot.nodes[0].total_gpus == 1

    assert [job.job_id for job in snapshot.queued_jobs] == [
        "high-priority-normal",
        "deadline-risk",
    ]

    jobs_by_id = {job.job_id: job for job in snapshot.queued_jobs}

    assert jobs_by_id["high-priority-normal"].priority == 15.0
    assert jobs_by_id["high-priority-normal"].deadline == 10
    assert jobs_by_id["high-priority-normal"].duration == 1

    assert jobs_by_id["deadline-risk"].priority == 6.0
    assert jobs_by_id["deadline-risk"].deadline == 3
    assert jobs_by_id["deadline-risk"].duration == 5


def test_build_value_density_snapshot_has_expected_shape() -> None:
    # Value-density snapshot:
    # 1 node, 3 jobs, 2 available GPUs.
    # Greedy should prefer the two dense 1-GPU jobs over the larger high-priority job.
    snapshot = build_value_density_snapshot()

    assert snapshot.current_time == 0
    assert len(snapshot.nodes) == 1
    assert len(snapshot.queued_jobs) == 3

    node = snapshot.nodes[0]

    assert node.node_id == "node-001"
    assert node.available_gpus == 2
    assert node.total_gpus == 4

    assert [job.job_id for job in snapshot.queued_jobs] == [
        "large-high-priority",
        "small-dense",
        "medium-dense",
    ]

    jobs_by_id = {job.job_id: job for job in snapshot.queued_jobs}

    assert jobs_by_id["large-high-priority"].gpu_demand == 4
    assert jobs_by_id["large-high-priority"].priority == 20.0

    assert jobs_by_id["small-dense"].gpu_demand == 1
    assert jobs_by_id["small-dense"].priority == 10.0

    assert jobs_by_id["medium-dense"].gpu_demand == 1
    assert jobs_by_id["medium-dense"].priority == 8.0

def test_build_capacity_limited_snapshot_has_expected_shape() -> None:
    # Capacity-limited snapshot:
    # 1 node, 3 jobs, 6 available GPUs.
    # Greedy must skip an oversized job and continue to a feasible one.
    snapshot = build_capacity_limited_snapshot()

    assert snapshot.current_time == 0
    assert len(snapshot.nodes) == 1
    assert len(snapshot.queued_jobs) == 3

    node = snapshot.nodes[0]

    assert node.node_id == "node-001"
    assert node.available_gpus == 6
    assert node.total_gpus == 8

    assert [job.job_id for job in snapshot.queued_jobs] == [
        "job-high-density",
        "job-large",
        "job-medium",
    ]

    jobs_by_id = {job.job_id: job for job in snapshot.queued_jobs}

    assert jobs_by_id["job-high-density"].gpu_demand == 2
    assert jobs_by_id["job-high-density"].priority == 10.0

    assert jobs_by_id["job-large"].gpu_demand == 7
    assert jobs_by_id["job-large"].priority == 14.0

    assert jobs_by_id["job-medium"].gpu_demand == 4
    assert jobs_by_id["job-medium"].priority == 8.0

def test_build_started_and_waiting_lateness_snapshot_has_expected_shape() -> None:
    # Started/waiting lateness snapshot:
    # 1 node, 2 jobs; one job is meant to start late, the other to wait.
    snapshot = build_started_and_waiting_lateness_snapshot()

    assert snapshot.current_time == 0
    assert len(snapshot.nodes) == 1
    assert len(snapshot.queued_jobs) == 2

    node = snapshot.nodes[0]

    assert node.node_id == "node-001"
    assert node.available_gpus == 4
    assert node.total_gpus == 4

    assert [job.job_id for job in snapshot.queued_jobs] == [
        "started-late",
        "waiting-risky",
    ]

    jobs_by_id = {job.job_id: job for job in snapshot.queued_jobs}

    assert jobs_by_id["started-late"].gpu_demand == 2
    assert jobs_by_id["started-late"].duration == 5
    assert jobs_by_id["started-late"].priority == 20.0
    assert jobs_by_id["started-late"].deadline == 3

    assert jobs_by_id["waiting-risky"].gpu_demand == 2
    assert jobs_by_id["waiting-risky"].duration == 3
    assert jobs_by_id["waiting-risky"].priority == 9.0
    assert jobs_by_id["waiting-risky"].deadline == 2

def test_build_random_snapshot_is_reproducible() -> None:
    # Random snapshot:
    # same seed and parameters must produce exactly the same jobs and nodes.
    snapshot_a = build_random_snapshot(
        num_nodes=3,
        num_jobs=5,
        seed=7,
    )

    snapshot_b = build_random_snapshot(
        num_nodes=3,
        num_jobs=5,
        seed=7,
    )

    assert snapshot_a == snapshot_b


def test_build_random_snapshot_has_expected_shape_and_bounds() -> None:
    # Random snapshot:
    # 3 nodes, 5 jobs, deterministic capacity and bounded generated fields.
    snapshot = build_random_snapshot(
        num_nodes=3,
        num_jobs=5,
        seed=11,
        gpu_demand_range=(1, 4),
        duration_range=(2, 9),
        priority_range=(5.0, 20.0),
        deadline_slack_range=(-2, 6),
        total_gpus_per_node=8,
        available_capacity_ratio=0.5,
    )

    assert snapshot.current_time == 0
    assert len(snapshot.nodes) == 3
    assert len(snapshot.queued_jobs) == 5

    assert [node.node_id for node in snapshot.nodes] == [
        "node-00000",
        "node-00001",
        "node-00002",
    ]

    assert all(node.total_gpus == 8 for node in snapshot.nodes)
    assert all(node.available_gpus == 4 for node in snapshot.nodes)

    assert [job.job_id for job in snapshot.queued_jobs] == [
        "job-00000",
        "job-00001",
        "job-00002",
        "job-00003",
        "job-00004",
    ]

    for job in snapshot.queued_jobs:
        assert 1 <= job.gpu_demand <= 4
        assert 2 <= job.duration <= 9
        assert 5.0 <= job.priority <= 20.0
        assert job.release_time == 0
        assert job.deadline >= job.release_time


def test_build_random_snapshot_rejects_invalid_parameters() -> None:
    # Random snapshot validation:
    # reject invalid scale, capacity, and range parameters before building schemas.
    invalid_cases = (
        {"num_nodes": 0, "num_jobs": 1, "seed": 1},
        {"num_nodes": 1, "num_jobs": -1, "seed": 1},
        {
            "num_nodes": 1,
            "num_jobs": 1,
            "seed": 1,
            "available_capacity_ratio": 1.5,
        },
        {
            "num_nodes": 1,
            "num_jobs": 1,
            "seed": 1,
            "gpu_demand_range": (4, 1),
        },
        {
            "num_nodes": 1,
            "num_jobs": 1,
            "seed": 1,
            "duration_range": (0, 2),
        },
        {
            "num_nodes": 1,
            "num_jobs": 1,
            "seed": 1,
            "priority_range": (-1.0, 2.0),
        },
    )

    for kwargs in invalid_cases:
        try:
            build_random_snapshot(**kwargs)
        except ValueError:
            continue

        raise AssertionError(f"Expected ValueError for parameters: {kwargs}")

