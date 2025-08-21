from typing import Dict, List, Tuple
from typing_extensions import TypeAlias

from collections import OrderedDict
import numpy as np

import cv2


Traj: TypeAlias = Dict[str, np.ndarray]     # 1 trajectory
Trajs: TypeAlias = List[Traj]               # List of trajectories
Trajz: TypeAlias = Tuple[np.ndarray, ...]   # Batch (Separated each component of concatenated trajectories)


"""
                    1 Step              1 Trajectory -- concat N trajs and separate --> Batch (total size B)
Observation         ob      [ob_dim,]   obs / observation           [ep_len,ob_dim]     observations        [B,ob_dim]   
Action              ac      [ac_dim,]   acs / action                [ep_len,ac_dim]     actions             [B,ac_dim]
Reward              rwd     scalar      rwds / reward               [ep_len,]           rewards             [B,]
Next Observation    next_ob [ob_dim,]   next_obs / next_observation [ep_len,ob_dim]     next_observations   [B,ob_dim]
Terminal            done    bool        dones / terminal            [ep_len,]           terminals           [B,]
"""


def sample_trajectory(env, policy, max_traj_length, render=False) -> Traj:
    """
    Sample a rollout in the environment from a policy.
    """
    # initialize env for the beginning of a new rollout
    ob, _ = env.reset()

    # initialize vars
    obs, acs, rwds, next_obs, dones, image_obs = [], [], [], [], [], []
    steps = 0

    while True:
        ac: np.ndarray = policy.get_action(ob)
        # ac = ac[0]    # policy class를 정의할 때 명시적으로 차원을 처리
        next_ob, rwd, done , *_ = env.step(ac)

        steps += 1
        rollout_done = True if (steps == max_traj_length or done) else False

        obs.append(ob)
        acs.append(ac)
        rwds.append(rwd)
        next_obs.append(next_ob)
        dones.append(rollout_done)

        # image_obs.append(image_ob)는 따로 처리
        if render:
            if hasattr(env, 'sim'):
                img = env.sim.render(camera_name='track', height=500, width=500)[::-1]
            else:
                img = env.render()

            image_obs.append(cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))

        ob = next_ob

        if rollout_done:
            break

    traj = {
        'observation': np.array(obs, dtype=np.float32),
        'action': np.array(acs, dtype=np.float32),
        'reward': np.array(rwds, dtype=np.float32),
        'next_observation': np.array(next_obs, dtype=np.float32),
        'terminal': np.array(dones, dtype=np.float32),
        'image_observation': np.array(image_obs, dtype=np.uint8)
    }

    return traj


def sample_trajectories(env, policy, min_timesteps_per_batch, max_traj_length, render=False) -> Tuple[Trajs, int]:
    """
    Collect rollouts until we have collected min_timesteps_per_batch steps.
    """
    trajs = []
    timesteps_this_batch = 0
    while timesteps_this_batch < min_timesteps_per_batch:
        traj = sample_trajectory(env, policy, max_traj_length, render)
        trajs.append(traj)

        timesteps_this_batch += get_traj_length(traj)

    return trajs, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_traj_length, render=False) -> Trajs:
    """
    Collect ntraj rollouts.
    """
    trajs = []
    for i in range(ntraj):
        traj = sample_trajectory(env, policy, max_traj_length, render)
        trajs.append(traj)

    return trajs


def convert_list_of_rollouts(trajs: Trajs, concat_rwds=True) -> Trajz:
    """
    Take a list of rollout dictionaries and return separate arrays,
    where each array is a concatenation of that array from across the rollouts.
    """
    observations = np.concatenate([traj['observation'] for traj in trajs])
    actions = np.concatenate([traj['action'] for traj in trajs])
    rewards = np.concatenate([traj['reward'] for traj in trajs]) if concat_rwds else [traj['reward'] for traj in trajs]
    next_observations = np.concatenate([traj['next_observation'] for traj in trajs])
    terminals = np.concatenate([traj['terminal'] for traj in trajs])
    # image_observation의 concatenation은 logger에서 처리

    trajz = observations, actions, rewards, next_observations, terminals

    return trajz


def compute_metrics(train_trajs: Trajs, eval_trajs: Trajs) -> Dict[str, float]:
    """
    Compute metrics for logging.
    """
    # returns, for logging
    train_returns = [train_traj['reward'].sum() for train_traj in train_trajs]
    eval_returns = [eval_traj['reward'].sum() for eval_traj in eval_trajs]

    # episode lengths, for logging
    train_ep_lens = [len(train_traj['reward']) for train_traj in train_trajs]
    eval_ep_lens = [len(eval_traj['reward']) for eval_traj in eval_trajs]

    # decide what to log
    logs = OrderedDict()

    logs['Eval_AverageReturn'] = np.mean(eval_returns)
    logs['Eval_StdReturn'] = np.std(eval_returns)
    logs['Eval_MaxReturn'] = np.max(eval_returns)
    logs['Eval_MinReturn'] = np.min(eval_returns)
    logs['Eval_AverageEpLen'] = np.mean(eval_ep_lens)

    logs['Train_AverageReturn'] = np.mean(train_returns)
    logs['Train_StdReturn'] = np.std(train_returns)
    logs['Train_MaxReturn'] = np.max(train_returns)
    logs['Train_MinReturn'] = np.min(train_returns)
    logs['Train_AverageEpLen'] = np.mean(train_ep_lens)

    return logs


def get_traj_length(traj: Traj):
    return len(traj['reward'])