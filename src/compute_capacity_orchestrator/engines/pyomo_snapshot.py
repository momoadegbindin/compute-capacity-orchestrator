"""Exact snapshot scheduling model using Pyomo."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from time import perf_counter

try:
    import pyomo.environ as pyo
except ImportError as exc:
    pyo = None
    _PYOMO_IMPORT_ERROR = exc
else:
    _PYOMO_IMPORT_ERROR = None


from compute_capacity_orchestrator.schemas.schedule import (
    GpuAssignment,
    SchedulingDecision,
    SchedulingSnapshot,
)

LOGGER = logging.getLogger(__name__)

def _ensure_pyomo_available() -> None:
    if pyo is None:
        raise RuntimeError(
            "Exact MIP scheduler requires Pyomo and HiGHS. "
            "Install pyomo and highspy, or select Greedy value density."
        ) from _PYOMO_IMPORT_ERROR

def _ensure_pyomo_snapshot_logger() -> None:
    """Attach a compact default console logger for Pyomo snapshot diagnostics."""

    if LOGGER.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

@dataclass(frozen=True, slots=True)
class PyomoModelStats:
    """Basic size counters for a Pyomo snapshot model."""

    num_jobs: int
    num_nodes: int
    num_assignment_variables: int
    num_start_variables: int
    num_variables: int
    num_full_demand_constraints: int
    num_node_capacity_constraints: int
    num_constraints: int

@dataclass(frozen=True, slots=True)
class PyomoSnapshotScheduler:
    """Exact MIP scheduler for a single scheduling snapshot."""

    deadline_penalty_weight: float = 0.0
    decision_step: int = 1
    solver_name: str = "appsi_highs"
    time_limit_seconds: float = 60
    relative_gap: float = .01
    log_diagnostics: bool = True

    def __post_init__(self) -> None:
        if self.deadline_penalty_weight < 0:
            raise ValueError("deadline_penalty_weight cannot be negative")

        if self.decision_step <= 0:
            raise ValueError("decision_step must be positive")

        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")

        if self.relative_gap is not None and self.relative_gap < 0:
            raise ValueError("relative_gap cannot be negative")

    def solve(self, snapshot: SchedulingSnapshot) -> SchedulingDecision:
        _ensure_pyomo_available()
        if self.log_diagnostics:
            _ensure_pyomo_snapshot_logger()
        total_start_time = perf_counter()

        model_build_start_time = perf_counter()
        model = self._build_model(snapshot)
        model_build_ms = (perf_counter() - model_build_start_time) * 1000.0

        model_stats = self._compute_model_stats(snapshot)

        solver_setup_start_time = perf_counter()

        solver = pyo.SolverFactory(self.solver_name)

        if not solver.available(exception_flag=False):
            raise RuntimeError(
                "Exact MIP scheduler requires the Pyomo Appsi HiGHS solver. "
                "Install pyomo and highspy, or select Greedy value density."
            )

        solver_setup_ms = (perf_counter() - solver_setup_start_time) * 1000.0
        if self.log_diagnostics:
            LOGGER.info(
                "\n[PyomoSnapshot] Model ready\n"
                "  jobs:                 %d\n"
                "  nodes:                %d\n"
                "  assignment variables: %d\n"
                "  start variables:      %d\n"
                "  total variables:      %d\n"
                "  constraints:          %d\n"
                "  model build:          %.3f ms\n"
                "  solver setup:         %.3f ms\n"
                "  solver:               %s",
                model_stats.num_jobs,
                model_stats.num_nodes,
                model_stats.num_assignment_variables,
                model_stats.num_start_variables,
                model_stats.num_variables,
                model_stats.num_constraints,
                model_build_ms,
                solver_setup_ms,
                self.solver_name,
            )

        solver_start_time = perf_counter()

        result = solver.solve(model,
            options={'time_limit': self.time_limit_seconds,
                     'mip_rel_gap': self.relative_gap})#, tee=True)

        solver_ms = (perf_counter() - solver_start_time) * 1000.0

        termination = result.solver.termination_condition

        if termination != pyo.TerminationCondition.optimal:
            raise RuntimeError(
                f"PyomoSnapshotScheduler did not solve to optimality: {termination}"
            )

        extraction_start_time = perf_counter()
        decision = self._extract_decision(
            snapshot=snapshot,
            model=model,
            runtime_ms=0.0,
        )
        extraction_ms = (perf_counter() - extraction_start_time) * 1000.0

        total_runtime_ms = (perf_counter() - total_start_time) * 1000.0
        decision = replace(decision, runtime_ms=total_runtime_ms)

        if self.log_diagnostics:
            LOGGER.info(
                "\n[PyomoSnapshot] Solve complete\n"
                "  termination:          %s\n"
                "  objective:            %.3f\n"
                "  model build:          %.3f ms\n"
                "  solver setup:         %.3f ms\n"
                "  solver call:          %.3f ms\n"
                "  extraction:           %.3f ms\n"
                "  total runtime:        %.3f ms",
                termination,
                decision.objective_value,
                model_build_ms,
                solver_setup_ms,
                solver_ms,
                extraction_ms,
                total_runtime_ms,
            )

        return decision

    def _compute_model_stats(self, snapshot: SchedulingSnapshot) -> PyomoModelStats:
        num_jobs = len(snapshot.queued_jobs)
        num_nodes = len(snapshot.nodes)

        num_assignment_variables = num_nodes * num_jobs
        num_start_variables = num_jobs

        num_full_demand_constraints = num_jobs
        num_node_capacity_constraints = num_nodes

        return PyomoModelStats(
            num_jobs=num_jobs,
            num_nodes=num_nodes,
            num_assignment_variables=num_assignment_variables,
            num_start_variables=num_start_variables,
            num_variables=num_assignment_variables + num_start_variables,
            num_full_demand_constraints=num_full_demand_constraints,
            num_node_capacity_constraints=num_node_capacity_constraints,
            num_constraints=num_full_demand_constraints + num_node_capacity_constraints,
        )

    def _build_model(self, snapshot: SchedulingSnapshot) -> pyo.ConcreteModel:
        model = pyo.ConcreteModel(name="snapshot_gpu_scheduling")

        job_ids = [job.job_id for job in snapshot.queued_jobs]
        node_ids = [node.node_id for node in snapshot.nodes]

        jobs_by_id = {job.job_id: job for job in snapshot.queued_jobs}
        nodes_by_id = {node.node_id: node for node in snapshot.nodes}

        gpu_demand = {
            job_id: jobs_by_id[job_id].gpu_demand for job_id in job_ids
        }

        priority = {
            job_id: jobs_by_id[job_id].priority for job_id in job_ids
        }

        available_gpus = {
            node_id: nodes_by_id[node_id].available_gpus for node_id in node_ids
        }

        started_lateness = {
            job_id: max(
                0,
                snapshot.current_time + jobs_by_id[job_id].duration
                - jobs_by_id[job_id].deadline,
            )
            for job_id in job_ids
        }

        waiting_lateness = {
            job_id: max(
                0,
                snapshot.current_time
                + self.decision_step
                + jobs_by_id[job_id].duration
                - jobs_by_id[job_id].deadline,
            )
            for job_id in job_ids
        }

        model.JOBS = pyo.Set(initialize=job_ids, ordered=True)
        model.NODES = pyo.Set(initialize=node_ids, ordered=True)

        model.x = pyo.Var(
            model.NODES,
            model.JOBS,
            domain=pyo.NonNegativeIntegers,
        )

        model.y = pyo.Var(
            model.JOBS,
            domain=pyo.Binary,
        )

        def full_gpu_demand_rule(model: pyo.ConcreteModel, job_id: str):
            return (
                sum(model.x[node_id, job_id] for node_id in model.NODES)
                == gpu_demand[job_id] * model.y[job_id]
            )

        model.full_gpu_demand = pyo.Constraint(
            model.JOBS,
            rule=full_gpu_demand_rule,
        )

        def node_capacity_rule(model: pyo.ConcreteModel, node_id: str):
            return (
                sum(model.x[node_id, job_id] for job_id in model.JOBS)
                <= available_gpus[node_id]
            )

        model.node_capacity = pyo.Constraint(
            model.NODES,
            rule=node_capacity_rule,
        )

        model.total_priority_started = pyo.Expression(
            expr=sum(
                priority[job_id] * model.y[job_id]
                for job_id in model.JOBS
            )
        )

        model.started_deadline_penalty = pyo.Expression(
            expr=sum(
                self.deadline_penalty_weight
                * started_lateness[job_id]
                * model.y[job_id]
                for job_id in model.JOBS
            )
        )

        model.waiting_deadline_penalty = pyo.Expression(
            expr=sum(
                self.deadline_penalty_weight
                * waiting_lateness[job_id]
                * (1 - model.y[job_id])
                for job_id in model.JOBS
            )
        )

        model.objective = pyo.Objective(
            expr=(
                model.total_priority_started
                - model.started_deadline_penalty
                - model.waiting_deadline_penalty
            ),
            sense=pyo.maximize,
        )

        return model

    def _extract_decision(
        self,
        snapshot: SchedulingSnapshot,
        model: pyo.ConcreteModel,
        runtime_ms: float,
    ) -> SchedulingDecision:
        assignments: list[GpuAssignment] = []
        started_job_ids: list[str] = []
        waiting_job_ids: list[str] = []

        for job in snapshot.queued_jobs:
            y_value = pyo.value(model.y[job.job_id])

            if y_value >= 0.5:
                started_job_ids.append(job.job_id)
            else:
                waiting_job_ids.append(job.job_id)

        for node in snapshot.nodes:
            for job in snapshot.queued_jobs:
                raw_value = pyo.value(model.x[node.node_id, job.job_id])
                gpu_count = int(round(raw_value))

                if gpu_count <= 0:
                    continue

                assignments.append(
                    GpuAssignment(
                        job_id=job.job_id,
                        node_id=node.node_id,
                        gpu_count=gpu_count,
                    )
                )

        return SchedulingDecision(
            assignments=tuple(assignments),
            started_job_ids=tuple(started_job_ids),
            waiting_job_ids=tuple(waiting_job_ids),
            objective_value=float(pyo.value(model.objective)),
            solver_status="optimal",
            runtime_ms=runtime_ms,
        )