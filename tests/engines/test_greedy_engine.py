from __future__ import annotations

from compute_capacity_orchestrator.engines.greedy import GreedyScheduler
from compute_capacity_orchestrator.engines.validation import validate_decision
from compute_capacity_orchestrator.schemas.schedule import GpuAssignment
from compute_capacity_orchestrator.experiments.scenarios import (
    build_empty_capacity_snapshot,
    build_split_node_snapshot,
    build_value_density_snapshot,
    build_capacity_limited_snapshot,
)

def test_greedy_scheduler_starts_no_jobs_when_capacity_is_empty() -> None:
    # Empty-capacity snapshot:
    # jobs are queued, but both nodes have zero available GPUs.
    snapshot = build_empty_capacity_snapshot()

    scheduler = GreedyScheduler()
    decision = scheduler.solve(snapshot)

    validate_decision(snapshot, decision)

    assert decision.started_job_ids == ()
    assert decision.waiting_job_ids == ("urgent-small", "batch-medium")
    assert decision.assignments == ()
    assert decision.objective_value == 0.0
    assert decision.solver_status == "greedy_value_density"
    assert decision.runtime_ms >= 0.0

def test_greedy_scheduler_returns_valid_decision() -> None:
    # Capacity-limited snapshot:
    # one node, 3 jobs, 6 available GPUs; greedy skips one oversized job and continues.
    snapshot = build_capacity_limited_snapshot()

    scheduler = GreedyScheduler()
    decision = scheduler.solve(snapshot)

    assert validate_decision(snapshot, decision) is None

    assert decision.started_job_ids == ("job-high-density", "job-medium")
    assert decision.waiting_job_ids == ("job-large",)
    assert decision.objective_value == 18.0
    assert decision.solver_status == "greedy_value_density"
    assert decision.runtime_ms >= 0.0

    assert len(decision.assignments) == 2
    assert decision.assignments[0].job_id == "job-high-density"
    assert decision.assignments[0].gpu_count == 2
    assert decision.assignments[1].job_id == "job-medium"
    assert decision.assignments[1].gpu_count == 4


def test_greedy_scheduler_uses_value_density_order() -> None:
    # Value-density snapshot:
    # one node, 3 jobs, 2 available GPUs; two dense jobs should start first.
    snapshot = build_value_density_snapshot()

    scheduler = GreedyScheduler()
    decision = scheduler.solve(snapshot)

    validate_decision(snapshot, decision)

    assert decision.started_job_ids == ("small-dense", "medium-dense")
    assert decision.waiting_job_ids == ("large-high-priority",)
    assert decision.objective_value == 18.0


def test_greedy_scheduler_splits_job_across_nodes_when_needed() -> None:
    # Split-node snapshot:
    # one 2-GPU job, two nodes with 1 available GPU each.
    snapshot = build_split_node_snapshot()

    scheduler = GreedyScheduler()
    decision = scheduler.solve(snapshot)

    validate_decision(snapshot, decision)

    assert decision.started_job_ids == ("multi-node-job",)
    assert decision.waiting_job_ids == ()
    assert decision.objective_value == 10.0

    assert decision.assignments == (
        GpuAssignment(
            job_id="multi-node-job",
            node_id="node-a",
            gpu_count=1,
        ),
        GpuAssignment(
            job_id="multi-node-job",
            node_id="node-b",
            gpu_count=1,
        ),
    )