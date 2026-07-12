EXPERIMENT_SIZE = "Small uses a fixed demo snapshot. Large generates a synthetic benchmark."
SCHEDULER = "Choose a fast heuristic or an exact optimization model."

NUM_NODES = "Number of GPU nodes available in the simulated cluster."
GPUS_PER_NODE = "GPU capacity available on each node."
NODE_AVAILABLE_GPUS = "Available GPU capacity on this demo node."

NUM_JOBS = "Number of queued jobs in the generated snapshot."
GPU_DEMAND_MIN = "Smallest GPU request generated for a job."
GPU_DEMAND_MAX = "Largest GPU request generated for a job."

HORIZON = "Number of simulation steps. Exact MIP solves once per step."
ARRIVAL_RATE = "Average number of new jobs arriving per simulation step."
SEED = "Controls reproducibility of the synthetic workload."

DEADLINE_WEIGHT = "Penalty strength for jobs that risk missing deadlines."
MIP_TIME_LIMIT = "Maximum solve time allowed for each optimization decision."
MIP_GAP = "Allowed optimality gap for faster MIP solves."