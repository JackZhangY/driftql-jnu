import os 

import d4rl
import gymnasium
import numpy as np
from d4rl.offline_env import set_dataset_path

from envs import D4RL_DATASET_DIR
from envs.env_utils import EpisodeMonitor
from utils.datasets import Dataset


def make_env(env_name):
    """Make D4RL environment."""
    env = gymnasium.make('GymV21Environment-v0', env_id=env_name)
    env = EpisodeMonitor(env)
    return env


def get_dataset(
    env,
    env_name,
    dataset_path=None,
):
    """Make D4RL dataset.

    Args:
        env: Environment instance.
        env_name: Name of the environment.
        dataset_path: Optional path to a custom dataset. Supports D4RL
            `.hdf5`/`.h5` files and `.npz` bundles. An `.npz` bundle is
            treated as qlearning format if it contains `next_observations`;
            otherwise it must be raw D4RL format with a `timeouts` field.
    """
    # Point D4RL's global cache at our directory so the default loader
    # (no explicit `dataset_path`) finds pre-downloaded files by URL basename.
    set_dataset_path(D4RL_DATASET_DIR)

    if dataset_path is None:
        dataset = d4rl.qlearning_dataset(env)
    else:
        dataset_path = os.path.expanduser(dataset_path)
        if dataset_path.endswith(('.hdf5', '.h5')):
            # Reuse D4RL's loader so infos/* fields and sanity checks are handled.
            dataset = d4rl.qlearning_dataset(env, h5path=dataset_path)
        elif dataset_path.endswith('.npz'):
            with np.load(dataset_path, allow_pickle=False) as bundle:
                raw = {key: bundle[key] for key in bundle.files}
            if 'next_observations' in raw:
                # Already in qlearning format.
                missing = [
                    key
                    for key in ('observations', 'actions', 'next_observations', 'terminals', 'rewards')
                    if key not in raw
                ]
                if missing:
                    raise KeyError(f'Dataset {dataset_path} is missing keys: {missing}')
                dataset = raw
            else:
                # Raw D4RL format; qlearning_dataset needs `timeouts` here
                # because the wrapped env does not expose `_max_episode_steps`.
                if 'timeouts' not in raw:
                    raise KeyError(f'Raw D4RL dataset {dataset_path} must contain a "timeouts" field.')
                dataset = d4rl.qlearning_dataset(env, dataset=raw)
        else:
            raise ValueError(f'Unsupported dataset format: {dataset_path}. Expected .hdf5, .h5, or .npz.')

    terminals = np.zeros_like(dataset['rewards'])  # Indicate the end of an episode.
    masks = np.zeros_like(dataset['rewards'])  # Indicate whether we should bootstrap from the next state.
    rewards = dataset['rewards'].copy().astype(np.float32)
    if 'antmaze' in env_name:
        for i in range(len(terminals) - 1):
            terminals[i] = float(
                np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
            )
            masks[i] = 1 - dataset['terminals'][i]
        rewards = rewards - 1.0
    else:
        for i in range(len(terminals) - 1):
            if (
                np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
                or dataset['terminals'][i] == 1.0
            ):
                terminals[i] = 1
            else:
                terminals[i] = 0
            masks[i] = 1 - dataset['terminals'][i]
    masks[-1] = 1 - dataset['terminals'][-1]
    terminals[-1] = 1

    return Dataset.create(
        observations=dataset['observations'].astype(np.float32),
        actions=dataset['actions'].astype(np.float32),
        next_observations=dataset['next_observations'].astype(np.float32),
        terminals=terminals.astype(np.float32),
        rewards=rewards,
        masks=masks,
    )
