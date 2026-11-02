"""Drift Field Policy (DFP).

Port of the reference implementation (DFP/agents/drift.py and
DFP/utils/drift_loss.py) adapted to this project's single-step transition
interface and network utilities.

Core idea
---------
The policy is a one-step pushforward map ``a = f_theta(s, z)`` with ``z ~ N(0, I)``.
Given a batch of states, we draw ``gen_per_label`` generated actions per state
and apply the drifting-field loss:

    targets = [generated (stop-grad), fixed positives (dataset actions)]
    affinity(x, y) ~ sqrt(softmax_x(-d(x,y)/R) * softmax_y(-d(x,y)/R))
    F(x) = sum_y coeff(x, y) * (y - x)

The positive part attracts generated actions toward dataset actions while the
implicit negative part (the generated samples themselves) repels them from each
other, yielding a multimodal one-step policy.  The resulting goal
``x + F(x)`` is detached and the actor regresses onto it (Wasserstein gradient
flow / drifting field).

For offline RL, a Q-loss on the same generated actions is added so that the
drift field can also be guided toward high-value regions (``q_alpha > 0``).
Set ``q_alpha = 0`` and ``alpha = 1`` to match the paper's offline
behavior-modeling stage.
"""

import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


def _cfg_get(cfg, key, default):
    try:
        return cfg[key]
    except KeyError:
        return default


def _cdist(x, y, eps=1e-8):
    """Pairwise L2 distance: [B, N, D] x [B, M, D] -> [B, N, M]."""
    xydot = jnp.einsum('bnd,bmd->bnm', x, y)
    xnorms = jnp.einsum('bnd,bnd->bn', x, x)
    ynorms = jnp.einsum('bmd,bmd->bm', y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return jnp.sqrt(jnp.clip(sq_dist, a_min=eps))


@partial(jax.jit, static_argnames=('R_list', 'plus_only', 'use_neg_only'))
def drift_loss(
    gen,
    fixed_pos,
    fixed_neg=None,
    weight_gen=None,
    weight_pos=None,
    weight_neg=None,
    R_list=(0.1,),
    plus_only=False,
    use_neg_only=False,
):
    """Drifting-field loss (ported from DFP/utils/drift_loss.py).

    Args:
        gen: [B, C_g, S] generated samples (gradients flow through this).
        fixed_pos: [B, C_p, S] positive (dataset) samples.
        fixed_neg: [B, C_n, S] optional explicit negative samples.
        weight_gen: [B, C_g] optional weights for generated samples.
        weight_pos: [B, C_p] optional weights for positive samples.
        weight_neg: [B, C_n] optional weights for negative samples.
        R_list: Kernel temperatures.
        plus_only: Drop the repulsive (negative) force.
        use_neg_only: Use only ``fixed_neg`` as negatives; generated samples are
            not treated as implicit negatives.

    Returns:
        loss: [B] per-state regression loss onto the detached drift goal.
        info: Dict with the distance scale and per-temperature force norms.
    """
    # 1. Defaults & casting.
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = jnp.zeros_like(gen[:, :0, :])
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = jnp.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = jnp.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = jnp.ones_like(fixed_neg[:, :, 0])

    gen = gen.astype(jnp.float32)
    fixed_pos = fixed_pos.astype(jnp.float32)
    fixed_neg = fixed_neg.astype(jnp.float32)
    weight_gen = weight_gen.astype(jnp.float32)
    weight_pos = weight_pos.astype(jnp.float32)
    weight_neg = weight_neg.astype(jnp.float32)

    old_gen = jax.lax.stop_gradient(gen)
    if use_neg_only:
        targets = jnp.concatenate([fixed_neg, fixed_pos], axis=1)
        targets_w = jnp.concatenate([weight_neg, weight_pos], axis=1)
    else:
        targets = jnp.concatenate([old_gen, fixed_neg, fixed_pos], axis=1)
        targets_w = jnp.concatenate([weight_gen, weight_neg, weight_pos], axis=1)

    # 2. Compute the detached drift goal.
    def calculate_scaled_goal_and_factor(old_gen_in, targets_in, targets_w_in):
        info = {}

        # Scale coordinates to have order-1 magnitude.
        dist = _cdist(old_gen_in, targets_in)
        weighted_dist = dist * targets_w_in[:, None, :]
        scale = weighted_dist.mean() / targets_w_in.mean()
        info['scale'] = scale

        scale_inputs = jnp.clip(scale / jnp.sqrt(S), a_min=1e-3)
        old_gen_scaled = old_gen_in / scale_inputs
        targets_scaled = targets_in / scale_inputs

        dist_normed = dist / jnp.clip(scale, a_min=1e-3)

        # Exclude self-interaction when generated samples are implicit negatives.
        if not use_neg_only:
            mask_val = 100.0
            diag_mask = jnp.eye(C_g, dtype=jnp.float32)
            block_mask = jnp.pad(diag_mask, ((0, 0), (0, C_n + C_p)))
            block_mask = jnp.expand_dims(block_mask, 0)
            dist_normed = dist_normed + block_mask * mask_val

        # Force loop over kernel temperatures.
        force_across_R = jnp.zeros_like(old_gen_scaled)
        for R in R_list:
            logits = -dist_normed / R

            affinity = jax.nn.softmax(logits, axis=-1)
            aff_transpose = jax.nn.softmax(logits, axis=-2)
            affinity = jnp.sqrt(jnp.clip(affinity * aff_transpose, a_min=1e-6))
            affinity = affinity * targets_w_in[:, None, :]

            split_idx = C_n if use_neg_only else C_g + C_n
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]

            sum_pos = jnp.sum(aff_pos, axis=-1, keepdims=True)
            r_coeff_neg = -aff_neg * sum_pos
            sum_neg = jnp.sum(aff_neg, axis=-1, keepdims=True)
            r_coeff_pos = aff_pos * sum_neg

            if plus_only:
                r_coeff_neg = jnp.zeros_like(r_coeff_neg)
            R_coeff = jnp.concatenate([r_coeff_neg, r_coeff_pos], axis=2)

            total_force_R = jnp.einsum('biy,byx->bix', R_coeff, targets_scaled)
            total_coeffs = R_coeff.sum(axis=-1)
            total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled

            f_norm_val = (total_force_R ** 2).mean()
            info[f'loss_{R}'] = f_norm_val

            # Normalize each temperature's force before summing.
            force_scale = jnp.sqrt(jnp.clip(f_norm_val, a_min=1e-8))
            force_across_R = force_across_R + total_force_R / force_scale

        goal_scaled = old_gen_scaled + force_across_R
        return goal_scaled, scale_inputs, info

    goal_scaled, scale_inputs, info = jax.lax.stop_gradient(
        calculate_scaled_goal_and_factor(old_gen, targets, targets_w)
    )
    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = jnp.mean(diff ** 2, axis=(-1, -2))
    info = jax.tree_util.tree_map(lambda x: x.mean(), info)

    return loss, info


class DFPAgent(flax.struct.PyTreeNode):
    """Drift Field Policy agent with a one-step actor and Q-learning."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _get_bon_samples(self, use_q_bon: bool):
        key = 'q_bon' if use_q_bon else 'eval_bon'
        n = _cfg_get(self.config, key, -1)
        if n is None or n <= 0:
            n = _cfg_get(self.config, 'actor_num_samples', 16)
        return int(n)

    def critic_loss(self, batch, grad_params, rng):
        """Standard TD critic loss with next actions from the one-step actor."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(
            batch['next_observations'], seed=sample_rng, use_q_bon=True
        )
        next_qs = self.network.select('target_critic')(
            batch['next_observations'], actions=next_actions
        )

        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        q = self.network.select('critic')(
            batch['observations'], actions=batch['actions'], params=grad_params
        )

        valid = batch['valid'] if 'valid' in batch else jnp.ones_like(target_q)
        critic_loss = (jnp.square(q - target_q) * valid).mean()

        info = {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'tgt_q_mean': next_q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }
        return critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Drift-field actor loss plus (optional) Q maximization."""
        batch_actions = batch['actions']
        batch_size, action_dim = batch_actions.shape
        rng, drift_rng = jax.random.split(rng)

        # Generate multiple samples per state from the current actor.
        gen_per_label = int(_cfg_get(self.config, 'gen_per_label', 8))
        obs_repeated = jnp.repeat(batch['observations'], gen_per_label, axis=0)
        drift_noises = jax.random.normal(
            drift_rng, (batch_size * gen_per_label, action_dim)
        )
        drift_actions_all = self.network.select('actor_drift')(
            obs_repeated, drift_noises, params=grad_params
        )
        drift_actions_all = jnp.clip(drift_actions_all, -1.0, 1.0)
        gen_samples = drift_actions_all.reshape(batch_size, gen_per_label, action_dim)

        # Q loss reuses the same generated actions (critic is stop-grad).
        qs_all = self.network.select('critic')(obs_repeated, actions=drift_actions_all)
        if self.config['q_agg'] == 'min':
            q_all = qs_all.min(axis=0)
        else:
            q_all = qs_all.mean(axis=0)
        q_all = q_all.reshape(batch_size, gen_per_label)

        q_loss = -q_all.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1.0 / (jnp.abs(q_all).mean() + 1e-6))
            q_loss = lam * q_loss
        q_loss = self.config['q_alpha'] * q_loss

        # Dataset actions are the fixed positive samples.
        pos_samples = jnp.expand_dims(batch_actions, axis=1)

        drift_temps = _cfg_get(self.config, 'drift_temps', (0.1,))
        if not isinstance(drift_temps, (tuple, list)):
            drift_temps = (drift_temps,)
        R_list = tuple(float(r) for r in drift_temps)
        drift_loss_val, drift_info = drift_loss(
            gen=gen_samples,
            fixed_pos=pos_samples,
            R_list=R_list,
            plus_only=bool(_cfg_get(self.config, 'drift_plus_only', False)),
        )

        alpha = float(self.config['alpha'])
        actor_drift_loss = alpha * drift_loss_val.mean()
        actor_loss = actor_drift_loss + q_loss

        info = {
            'actor_loss': actor_loss,
            'actor_drift_loss': actor_drift_loss,
            'drift_scale': drift_info.get('scale', 0.0),
            'q_loss': q_loss,
            'q_gen_mean': q_all.mean(),
        }
        for key, val in drift_info.items():
            if key.startswith('loss_'):
                info[f'drift_{key}'] = val
        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    def _score_actions(self, observations, actions):
        qs = self.network.select('critic')(observations, actions=actions)
        if self.config['q_agg'] == 'mean':
            return qs.mean(axis=0)
        return qs.min(axis=0)

    def _select_best_bon_action(self, actions, q_values):
        indices = jnp.argmax(q_values, axis=-1)
        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        return jnp.reshape(actions, (-1, actions.shape[-2], actions.shape[-1]))[
            jnp.arange(bsize), indices, :
        ].reshape(bshape + (actions.shape[-1],))

    @partial(jax.jit, static_argnames=('use_q_bon',))
    def sample_actions(self, observations, seed=None, temperature=1.0, use_q_bon=False):
        """Sample actions from the one-step actor (optionally best-of-n)."""
        del temperature  # Kept for the project-wide sample_actions interface.
        if seed is None:
            seed = jax.random.PRNGKey(0)

        action_dim = self.config['action_dim']
        batch_dims = observations.shape[: -len(self.config['ob_dims'])]
        actor_type = _cfg_get(self.config, 'actor_type', 'distill-ddpg')

        if actor_type == 'best-of-n':
            num_samples = self._get_bon_samples(use_q_bon=use_q_bon)
            seed, init_noise_rng = jax.random.split(seed)
            noises = jax.random.normal(
                init_noise_rng, (*batch_dims, num_samples, action_dim)
            )
            obs_rep = jnp.repeat(observations[..., None, :], num_samples, axis=-2)
            actions = self.network.select('actor_drift')(obs_rep, noises)
            actions = jnp.clip(actions, -1.0, 1.0)
            actions = jnp.where(jnp.isnan(actions), 0.0, actions)

            q = self._score_actions(obs_rep, actions)
            return self._select_best_bon_action(actions, q)

        noises = jax.random.normal(seed, (*batch_dims, action_dim))
        actions = self.network.select('actor_drift')(observations, noises)
        actions = jnp.clip(actions, -1.0, 1.0)
        return jnp.where(jnp.isnan(actions), 0.0, actions)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_drift'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=_cfg_get(config, 'num_qs', 2),
            encoder=encoders.get('critic'),
        )
        actor_drift_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_drift'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_drift=(actor_drift_def, (ex_observations, ex_actions)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        if _cfg_get(config, 'actor_type', None) is None:
            config['actor_type'] = 'distill-ddpg'
        if _cfg_get(config, 'actor_num_samples', None) is None:
            config['actor_num_samples'] = 16

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dfp',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='mean',  # "mean" or "min"
            num_qs=2,
            # Drift-field loss.
            gen_per_label=8,  # Generated samples per state for the drift loss.
            drift_temps=(0.1,),  # Kernel temperatures R.
            alpha=1.0,  # Drift loss weight.
            drift_plus_only=False,  # Drop the repulsive force when True.
            # Q-learning.
            q_alpha=1.0,  # Q-loss weight (0 reproduces the paper's offline BC stage).
            normalize_q_loss=True,
            # Evaluation / target action sampling.
            actor_type='distill-ddpg',  # "distill-ddpg" or "best-of-n"
            actor_num_samples=16,
            q_bon=-1,  # <= 0 falls back to actor_num_samples.
            eval_bon=-1,  # <= 0 falls back to actor_num_samples.
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
