"""Run a small snapshot scheduling demo."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from compute_capacity_orchestrator.engines.greedy import GreedyScheduler
from compute_capacity_orchestrator.engines.validation import validate_decision
from compute_capacity_orchestrator.experiments.scenarios import build_small_snapshot
from compute_capacity_orchestrator.metrics.decision_metrics import compute_decision_metrics


def main() -> None:
    snapshot = build_small_snapshot()
    scheduler = GreedyScheduler()
    deadline_penalty_weight = 0.0
    decision = scheduler.solve(snapshot)
    validate_decision(snapshot, decision)

    metrics = compute_decision_metrics(
        snapshot=snapshot,
        decision=decision,
        deadline_penalty_weight=deadline_penalty_weight,
    )

    print("\n=== Compute Capacity Orchestrator: Snapshot Demo ===\n")

    print("Queued jobs:")
    for job in snapshot.queued_jobs:
        print(
            f"  {job.job_id:20s} "
            f"gpus={job.gpu_demand:<2d} "
            f"duration={job.duration:<2d} "
            f"priority={job.priority:<5.1f} "
            f"deadline={job.deadline}"
        )

    print("\nCluster nodes:")
    for node in snapshot.nodes:
        print(
            f"  {node.node_id:8s} "
            f"available_gpus={node.available_gpus:<2d} "
            f"total_gpus={node.total_gpus:<2d} "
            f"group={node.topology_group}"
        )

    print("\nAssignments:")
    for assignment in decision.assignments:
        print(
            f"  job={assignment.job_id:20s} "
            f"node={assignment.node_id:8s} "
            f"gpus={assignment.gpu_count}"
        )

    print("\nDecision:")
    print(f"  started_jobs = {list(decision.started_job_ids)}")
    print(f"  waiting_jobs = {list(decision.waiting_job_ids)}")
    print(f"  reported_objective    = {decision.objective_value:.2f}")
    print(f"  status       = {decision.solver_status}")
    print(f"  runtime_ms   = {decision.runtime_ms:.3f}")

    print("\nMetrics:")
    print(f"  jobs_submitted          = {metrics.jobs_submitted}")
    print(f"  jobs_started            = {metrics.jobs_started}")
    print(f"  jobs_waiting            = {metrics.jobs_waiting}")
    print(f"  gpu_capacity_available  = {metrics.gpu_capacity_available}")
    print(f"  gpu_capacity_used       = {metrics.gpu_capacity_used}")
    print(f"  gpu_utilization         = {metrics.gpu_utilization:.2%}")
    print(f"  total_priority_started  = {metrics.total_priority_started:.2f}")
    print(f"  started_deadline_penalty = {metrics.started_deadline_penalty:.2f}")
    print(f"  waiting_deadline_penalty = {metrics.waiting_deadline_penalty:.2f}")
    print(f"  deadline_penalty        = {metrics.deadline_penalty_incurred:.2f}")


if __name__ == "__main__":
    main()