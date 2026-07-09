"""Run greedy snapshot scheduling on generated scale scenarios."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from compute_capacity_orchestrator.engines.greedy import GreedyScheduler
from compute_capacity_orchestrator.engines.pyomo_snapshot import PyomoSnapshotScheduler
from compute_capacity_orchestrator.engines.validation import validate_decision
from compute_capacity_orchestrator.experiments.scenarios import (build_random_snapshot,build_greedy_vs_mip_snapshot)
from compute_capacity_orchestrator.metrics.decision_metrics import (
    compute_decision_metrics,
)

DEADLINE_PENALTY_WEIGHT = 1.0
DECISION_STEP = 1
GREEDY = False # True
#GREEDY = True
CASES = (
    #("small", 20, 98, 101),
    #("small", 15, 11, 27),
    #("medium", 100, 1_000, 112),
    ("large", 1_000, 10_00, 103),
    # Uncomment only when you intentionally want a stress run.
    # ("stress", 10_000, 50_000, 104),
)


def run_case(
    name: str,
    num_nodes: int,
    num_jobs: int,
    seed: int,
) -> dict[str, object]:

    snapshot = build_random_snapshot(
        num_nodes=num_nodes,
        num_jobs=num_jobs,
        seed=seed,
        gpu_demand_range=(1, 8),
        duration_range=(1, 24),
        priority_range=(1.0, 100.0),
        deadline_slack_range=(-4, 24),
        total_gpus_per_node=8,
        available_capacity_ratio=0.70,
    )

    scheduler = GreedyScheduler() if GREEDY else PyomoSnapshotScheduler(
        deadline_penalty_weight=DEADLINE_PENALTY_WEIGHT,
        decision_step=DECISION_STEP,
    )
    decision = scheduler.solve(snapshot)

    validate_decision(snapshot, decision)

    metrics = compute_decision_metrics(
        snapshot=snapshot,
        decision=decision,
        deadline_penalty_weight=DEADLINE_PENALTY_WEIGHT,
        decision_step=DECISION_STEP,
    )

    return {
        "case": name,
        "nodes": len(snapshot.nodes),
        "jobs": len(snapshot.queued_jobs),
        "started": f"{metrics.jobs_started}/{metrics.jobs_submitted}",
        "gpu_use": f"{metrics.gpu_capacity_used}/{metrics.gpu_capacity_available}",
        "utilization": f"{metrics.gpu_utilization:.1%}",
        "scheduler_objective": round(decision.objective_value, 1),
        "objective": round(metrics.objective_value, 1),
        "scheduler_ms": round(decision.runtime_ms, 3),
    }


def main() -> None:
    rows = [run_case(*case) for case in CASES]

    print("\n=== Snapshot Scale Demo ===\n")
    if GREEDY:
        print("Scheduler: Greedy value density\n")
    else:
        print("Scheduler: Pyomo with HiGHS\n")

    header = (
        f"{'case':<10}"
        f"{'nodes':>10}"
        f"{'jobs':>10}"
        f"{'started':>14}"
        f"{'gpu use':>16}"
        f"{'util':>10}"
        f"{'objective':>14}"
        f"{'sched ms':>12}"
    )

    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['case']:<10}"
            f"{row['nodes']:>10}"
            f"{row['jobs']:>10}"
            f"{row['started']:>14}"
            f"{row['gpu_use']:>16}"
            f"{row['utilization']:>10}"
            f"{row['objective']:>14}"
            f"{row['scheduler_ms']:>12}"
        )

    print()


if __name__ == "__main__":
    main()