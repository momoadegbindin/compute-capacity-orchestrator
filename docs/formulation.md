# Formulation

This document describes the optimization models behind Compute Capacity Orchestrator.

The current implementation includes a single-decision snapshot MIP. Future versions extend the same decision structure to time-indexed planning, rolling-horizon control, topology-aware placement, stochastic scenarios, and decomposition-based models.

## 1. Implemented Snapshot MIP

At scheduling time $t_0$, the scheduler observes a set of eligible queued jobs and currently available GPU capacity. The decision is which jobs start now and how many GPUs each selected job receives from each node.

### Sets

$$
J = \text{set of eligible queued jobs}

$$

$$
N = \text{set of available cluster nodes}

$$

### Parameters

For each job $j \in J$:

$$
g_j = \text{GPU demand}

$$

$$
d_j = \text{duration}

$$

$$
p_j = \text{priority or value}

$$

$$
\ell_j = \text{deadline}

$$

For each node $i \in N$:

$$
c_i = \text{currently available GPUs}

$$

The current scheduling time is $t_0$. The decision step used to estimate one-period waiting impact is $\Delta$. The deadline penalty weight is $\beta \ge 0$.

### Derived lateness terms

If job $j$ starts now:

$$
L_j^{start} = \max(0, t_0 + d_j - \ell_j)

$$

If job $j$ waits at least one decision step:

$$
L_j^{wait} = \max(0, t_0 + \Delta + d_j - \ell_j)

$$

### Decision variables

$$
x_{ij} \in \mathbb{Z}_+

$$

number of GPUs from node $i$ assigned to job $j$.

$$
y_j \in \{0,1\}

$$

equals 1 if job $j$ starts now, and 0 otherwise.

### Objective

$$
\max
\sum_{j \in J} p_j y_j
-
\beta \sum_{j \in J} L_j^{start} y_j
-
\beta \sum_{j \in J} L_j^{wait}(1-y_j)

$$

The first term rewards starting high-value work. The second term penalizes jobs that start but still finish late. The third term penalizes jobs whose waiting decision increases deadline risk.

When $\beta = 0$, the model reduces to maximizing started priority subject to GPU capacity.

### Constraints

A selected job must receive its full GPU demand:

$$
\sum_{i \in N} x_{ij} = g_j y_j
\qquad \forall j \in J

$$

Node capacity cannot be exceeded:

$$
\sum_{j \in J} x_{ij} \le c_i
\qquad \forall i \in N

$$

Variable domains:

$$
x_{ij} \in \mathbb{Z}_+
\qquad \forall i \in N, j \in J

$$

$$
y_j \in \{0,1\}
\qquad \forall j \in J

$$

## 2. Operational Decision Budget

The model is used as a scheduling decision engine, so solution time matters.

The current exact snapshot scheduler supports solver time limits and MIP-gap controls. These controls are part of the operational contract: the scheduler must return a decision quickly enough to be useful inside an interactive dashboard or a repeated simulation loop.

The planned hybrid scheduler will make this budget explicit. A fast heuristic will provide a feasible baseline quickly. The exact solver may then improve the solution within a bounded time window. If the solver times out, fails, or cannot improve the incumbent, the system can safely fall back to the heuristic decision.

## 3. Current Modeling Scope

In the current model, we made the following assumptions:

* the model sees only the current queue and available capacity,
* job durations are treated as known,
* GPU demand is represented as a count,
* assignments may be split across nodes,
* topology costs and GPU-type compatibility are not yet part of the objective,
* the model does not reserve capacity for future arrivals,
* running jobs are represented only through reduced available capacity.

## 4. Planned Formulation Layers

### Time-indexed scheduling

The next model extends the snapshot formulation over a planning horizon. Instead of deciding only which jobs start now, the scheduler decides which jobs start at each future time period.

A time-indexed model introduces variables such as:

$$
y_{jt} = 1

$$

if job $j$ starts at time $t$, and

$$
x_{ijt}

$$

for GPUs assigned from node $i$ to job $j$ at time $t$.

This allows the scheduler to reason about future capacity, release times, deadlines, and rolling-horizon control.

### Rolling-horizon control

A rolling-horizon scheduler repeatedly solves a short planning problem, executes only the first decision, observes the new state, and replans. This connects the optimization model to the simulation loop.

### Topology-aware placement

Future placement models will account for GPU type, node groups, rack locality, and network-distance penalties. These terms can be added without changing the external scheduler contract.

### Fragmentation metrics and penalties

Capacity fragmentation occurs when the cluster has idle GPUs but those GPUs are scattered in a way that blocks larger jobs.

A future metric can track whether a job of size $k$ is blocked despite enough aggregate idle capacity:

$$
\sum_i c_i \ge k \quad \text{and} \quad \max_i c_i < k

$$

This is a simple same-node version of the fragmentation signal. Later versions can generalize it to rack locality, topology groups, GPU types, and multi-node placement patterns.

### Stochastic and robust formulations

Future stochastic models will evaluate scheduling policies across multiple possible arrival paths, capacity shocks, and workload-mix changes. Robust variants will penalize policies that perform well on average but fail badly under demand spikes or capacity loss.

### Decomposition-based models

Large time-indexed and topology-aware models may become too large for monolithic MIP solves. Planned research directions include column generation for placement-pattern models and Benders-style decomposition for planning, recourse, and feasibility separation.

## 5. Solver Backends

The current exact snapshot implementation uses Pyomo with HiGHS.

Future solver experiments may use JuMP with HiGHS or commercial solvers such as Gurobi for larger time-indexed, decomposition, and stochastic formulations. The goal is to keep the external scheduler contract stable while allowing the internal optimization backend to evolve.
