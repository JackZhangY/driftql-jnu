"""Environment and dataset utilities."""

import os

# Directory containing locally cached D4RL datasets. D4RL looks up files by the
# basename of their dataset URL, so pre-downloaded files can be used directly.
# Override with D4RL_DATASET_DIR.
D4RL_DATASET_DIR = os.environ.get(
    # 'D4RL_DATASET_DIR', os.path.expanduser('/root/autodl-tmp/d4rl/datasets') # run on autoDL
    'D4RL_DATASET_DIR', os.path.expanduser('~/.d4rl/datasets') # run on local machine
)

# Directory used by OGBench's loader (train and validation splits). Missing
# datasets are downloaded here automatically. Change the default below or set
# OGBENCH_DATASET_DIR.
OGBENCH_DATASET_DIR = os.environ.get(
    # 'OGBENCH_DATASET_DIR', os.path.expanduser('/root/autodl-tmp/ogbench/data') # run on autoDL 
    'OGBENCH_DATASET_DIR', os.path.expanduser('~/.ogbench/data') # run on local machine
)