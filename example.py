import os
from tempfile import NamedTemporaryFile
from launcher import (
    ComputeNodeManager, 
    InsufficientResources,
    MPIRun,
)
import time

# Commands that run in a sub-shell immediately before each task:
MPIRun.set_preamble_commands(
    'eval "$(/lus/theta-fs0/projects/RL-fold/msalim/miniconda3/bin/conda shell.bash hook)"',
    'conda activate /lus/theta-fs0/projects/RL-fold/venkatv/software/conda_env/a100_rapids_openmm',
)


node_manager = ComputeNodeManager()
num_nodes = len(node_manager.nodes)

runs = []
os.makedirs("test-outputs", exist_ok=True)

for i in range(num_nodes):
    nodes, gpus = node_manager.request(num_nodes=1, gpus_per_node=1)

    output_file = NamedTemporaryFile(dir="./test-outputs", delete=False)

    run = MPIRun(
        cmd_line="python hello.py",
        node_list=nodes,
        ranks_per_node=1,
        gpu_ids=gpus,
        output_file=output_file,
        cwd=None, # can be a different working directory
        envs_dict=None, # can be a dictionary of environ vars to add
    )
    runs.append(run)

while runs:
    # Clean up runs which finished (poll() returns process code)
    print("waiting on", len(runs), "runs to finish...")
    runs = [run for run in runs if run.poll() is None]
    time.sleep(5)

print("All done!")
