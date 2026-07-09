from __future__ import annotations

from compute_capacity_orchestrator.metrics.decision_metrics import (
    compute_decision_metrics,
)
from compute_capacity_orchestrator.schemas.resources import ClusterNode
from compute_capacity_orchestrator.schemas.schedule import (
    GpuAssignment,
    SchedulingDecision,
    SchedulingSnapshot,
)
from compute_capacity_orchestrator.schemas.workload import JobRequest

from app.views.snapshot_components import (
    build_assignments_dataframe,
    build_capacity_dataframe,
    build_jobs_dataframe,
    build_nodes_dataframe,
    build_run_history_row,
)


def _snapshot() -> SchedulingSnapshot:
    return SchedulingSnapshot(
        current_time=0,
        queued_jobs=(
            JobRequest(
                job_id="job-a",
                gpu_demand=2,
                duration=5,
                priority=10.0,
                release_time=0,
                deadline=20,
            ),
            JobRequest(
                job_id="job-b",
                gpu_demand=1,
                duration=3,
                priority=5.0,
                release_time=0,
                deadline=10,
            ),
        ),
        nodes=(
            ClusterNode(
                node_id="node-001",
                total_gpus=4,
                available_gpus=4,
            ),
        ),
    )


def _decision() -> SchedulingDecision:
    return SchedulingDecision(
        assignments=(
            GpuAssignment(
                job_id="job-a",
                node_id="node-001",
                gpu_count=2,
            ),
        ),
        started_job_ids=("job-a",),
        waiting_job_ids=("job-b",),
        objective_value=10.0,
        solver_status="manual",
        runtime_ms=1.2345,
    )


def test_build_jobs_dataframe_labels_started_and_waiting_jobs() -> None:
    jobs_df = build_jobs_dataframe(
        snapshot=_snapshot(),
        decision=_decision(),
    )

    decisions_by_job = dict(zip(jobs_df["job_id"], jobs_df["decision"]))

    assert decisions_by_job == {
        "job-a": "start",
        "job-b": "wait",
    }


def test_build_nodes_dataframe_contains_cluster_capacity() -> None:
    nodes_df = build_nodes_dataframe(_snapshot())

    assert list(nodes_df.columns) == [
        "node_id",
        "available_gpus",
        "total_gpus",
        "topology_group",
    ]
    assert nodes_df.loc[0, "node_id"] == "node-001"
    assert nodes_df.loc[0, "available_gpus"] == 4
    assert nodes_df.loc[0, "total_gpus"] == 4


def test_build_assignments_dataframe_preserves_assignment_rows() -> None:
    assignments_df = build_assignments_dataframe(_decision())

    assert list(assignments_df.columns) == [
        "job_id",
        "node_id",
        "gpu_count",
    ]
    assert assignments_df.loc[0, "job_id"] == "job-a"
    assert assignments_df.loc[0, "node_id"] == "node-001"
    assert assignments_df.loc[0, "gpu_count"] == 2


def test_build_assignments_dataframe_has_columns_when_empty() -> None:
    decision = SchedulingDecision(
        assignments=(),
        started_job_ids=(),
        waiting_job_ids=("job-a", "job-b"),
        objective_value=0.0,
        solver_status="manual",
        runtime_ms=0.0,
    )

    assignments_df = build_assignments_dataframe(decision)

    assert assignments_df.empty
    assert list(assignments_df.columns) == [
        "job_id",
        "node_id",
        "gpu_count",
    ]


def test_build_capacity_dataframe_computes_used_and_unused_gpus() -> None:
    snapshot = _snapshot()
    assignments_df = build_assignments_dataframe(_decision())

    capacity_df = build_capacity_dataframe(
        snapshot=snapshot,
        assignments_df=assignments_df,
    )

    assert capacity_df.loc[0, "node_id"] == "node-001"
    assert capacity_df.loc[0, "used_gpus"] == 2
    assert capacity_df.loc[0, "unused_gpus"] == 2


def test_build_run_history_row_uses_metrics_and_snapshot_capacity() -> None:
    snapshot = _snapshot()
    decision = _decision()

    metrics = compute_decision_metrics(
        snapshot=snapshot,
        decision=decision,
        deadline_penalty_weight=0.0,
        decision_step=1,
    )

    row = build_run_history_row(
        snapshot=snapshot,
        metrics=metrics,
        scheduler_name="Greedy value density",
        scenario_name="small-capacity",
        time_label="12:00:00",
    )

    assert row["time"] == "12:00:00"
    assert row["scheduler"] == "Greedy value density"
    assert row["scenario"] == "small-capacity"
    assert row["jobs"] == "1/2"
    assert row["cluster"] == "1 nodes, 4/4 GPUs available"
    assert row["gpu use"] == "2/4"
    assert row["utilization"] == "50.0%"
    assert row["objective"] == 10.0
    assert row["runtime_ms"] == 1.234