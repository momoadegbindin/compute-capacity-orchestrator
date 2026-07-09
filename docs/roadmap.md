# Roadmap

## Phase 1 - Core harness

Current public release.

- Typed schemas for jobs, resources, topology, snapshots, assignments, and scheduling decisions
- Queue state and cluster state
- Deterministic workload and cluster scenarios
- Heavy-tailed synthetic arrivals for simulation
- Greedy snapshot scheduler
- Exact snapshot MIP scheduler using Pyomo/HiGHS
- Decision validation
- State updates for waiting, running, and completed jobs
- Closed-loop simulation
- Metrics for utilization, queue length, wait time, deadline risk, objective value, and scheduler runtime
- Streamlit dashboard with snapshot and simulation views
- Regression tests for schemas, engines, validation, metrics, scenarios, simulation, and dashboard data builders

## Phase 2 - Exact optimization

- Time-indexed MIP formulation over a short planning horizon
- Rolling-horizon scheduling
- GPU capacity constraints over time
- Release-time constraints
- Deadline-aware planning
- Reference solver integrations

## Phase 3 - Hybrid scheduling

- Bounded solver runtimes
- Fallback policies
- Solver-status handling
- Gap and timeout controls
- Comparison of solution quality against scheduler latency

## Phase 4 - Topology and workload realism

- GPU types 
- Node groups
- Multi-node placement
- H100-class accelerator pools
- NVLink domains
- InfiniBand/Ethernet fabric effects,
- Rack locality
- Topology penalties
- Workload classes such as inference, evaluation, simulation, and training

## Phase 5 - Policy laboratory

- Stochastic demand scenarios
- Capacity drops
- Quota policies
- Admission-control experiments
- Cloud-burst options
- Dashboard-based policy comparison

## Phase 6 - Advanced optimization research

- Column generation for large placement-pattern models
- Benders-style decomposition for planning, recourse, and feasibility separation
- Stochastic recourse formulations
- Larger solver experiments with alternative modeling backends