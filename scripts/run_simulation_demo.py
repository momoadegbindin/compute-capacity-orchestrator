"""Run a stochastic discrete-time GPU scheduling simulation demo."""

from __future__ import annotations

import sys
from pathlib import Path

from compute_capacity_orchestrator.engines.pyomo_snapshot import PyomoSnapshotScheduler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from compute_capacity_orchestrator.engines.greedy import GreedyScheduler
from compute_capacity_orchestrator.schemas.resources import ClusterNode
from compute_capacity_orchestrator.simulation.arrivals import (
    generate_heavy_tailed_arrivals_by_time,
)
from compute_capacity_orchestrator.simulation.simulation_loop import (
    compute_simulation_summary,
    run_deterministic_simulation,
)
from compute_capacity_orchestrator.simulation.state import SimulationState


def build_initial_state() -> SimulationState:
    """Build the initial cluster state for the simulation demo."""

    nodes = tuple(
        ClusterNode(
            node_id=f"node-{node_index:03d}",
            total_gpus=8,
            available_gpus=8,
            topology_group=f"rack-{node_index // 4}",
        )
        for node_index in range(8)
    )

    return SimulationState(
        current_time=0,
        waiting_jobs=(),
        running_jobs=(),
        nodes=nodes,
        completed_jobs=(),
    )


def main() -> None:
    horizon = 50
    arrival_rate = 3.0
    seed = 42

    initial_state = build_initial_state()

    arrivals_by_time = generate_heavy_tailed_arrivals_by_time(
        start_time=0,
        horizon=horizon,
        seed=seed,
        arrival_rate=arrival_rate,
        gpu_demand_range=(1, 16),
        duration_range=(1, 12),
        priority_range=(1.0, 100.0),
        deadline_slack_range=(-2, 12),
        max_arrivals_per_step=12,
    )

    #scheduler = GreedyScheduler()
    scheduler = PyomoSnapshotScheduler(deadline_penalty_weight=1.0)
    result = run_deterministic_simulation(
        initial_state=initial_state,
        scheduler=scheduler,
        arrivals_by_time=arrivals_by_time,
        horizon=horizon,
        deadline_penalty_weight=1.0,
        decision_step=1,
    )

    summary = compute_simulation_summary(result)

    print("\n=== Simulation Demo ===\n")
    print("Simulation type: Discrete-time stochastic arrivals")
    print("Scheduler:       Greedy value density")
    print(f"Horizon:         {horizon} time steps")
    print(f"Arrival rate:    {arrival_rate:.2f} jobs / step")
    print(f"Seed:            {seed}")
    print(f"Cluster:         {len(initial_state.nodes)} nodes, 8 GPUs each")
    print()

    print("=== Aggregate Summary ===\n")
    print(f"Jobs arrived:              {summary.total_jobs_arrived}")
    print(f"Jobs completed:            {summary.total_jobs_completed}")
    print(f"Jobs waiting at end:        {summary.jobs_waiting_final}")
    print(f"Jobs running at end:        {summary.jobs_running_final}")
    print(f"Average GPU utilization:    {summary.average_gpu_utilization:.1%}")
    print(f"Average queue length:       {summary.average_queue_length:.2f}")
    print(f"Average scheduler runtime:  {summary.average_scheduler_runtime_ms:.3f} ms")
    print(f"Total priority completed:   {summary.total_priority_completed:.1f}")
    print(f"Deadline misses:            {summary.deadline_miss_count}")
    print(f"Deadline miss rate:         {summary.deadline_miss_rate:.1%}")
    print(f"Average completed wait:     {summary.average_completed_wait_time:.2f}")
    print()

    print("=== Step Metrics ===\n")

    header = (
        f"{'time':>4}"
        f"{'arr':>6}"
        f"{'start':>8}"
        f"{'wait':>8}"
        f"{'run':>8}"
        f"{'done':>8}"
        f"{'gpu':>10}"
        f"{'util':>8}"
        f"{'obj':>12}"
        f"{'ms':>10}"
    )

    print(header)
    print("-" * len(header))

    for step in result.step_metrics:
        print(
            f"{step.time:>4}"
            f"{step.jobs_arrived:>6}"
            f"{step.jobs_started:>8}"
            f"{step.jobs_waiting:>8}"
            f"{step.jobs_running:>8}"
            f"{step.jobs_completed:>8}"
            f"{step.gpu_capacity_used:>5}/{step.gpu_capacity_total:<4}"
            f"{step.gpu_utilization:>8.1%}"
            f"{step.objective_value:>12.1f}"
            f"{step.scheduler_runtime_ms:>10.3f}"
        )

    print()


if __name__ == "__main__":
    main()