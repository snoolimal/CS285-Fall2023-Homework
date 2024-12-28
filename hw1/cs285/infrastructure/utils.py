from typing import TypeAlias, List, Dict, Tuple
from collections import OrderedDict
import numpy as np
import cv2

from torch import nn

import gymnasium as gym


Traj: TypeAlias = Dict[str, np.ndarray]
Trajs: TypeAlias = List[Traj]
TrajInfo: TypeAlias = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def sample_trajectory(
    env: gym.Env, policy: nn.Module, max_traj_length: int, render: bool = False
) -> Traj:
    """
    Current policy를 env에 굴려 rollout 1개를 sampling한다.
    """
    obs, acs, rwds, next_obs, terms, image_obs = [], [], [], [], [], []

    steps = 0
    ob, _ = env.reset()     # ob의 shape은 env.observation_space.shape
    while True:
        if render:
            if hasattr(env, 'sim'):
                img = env.sim.render(camera_name='track', height=500, width=500)[::-1]
            else:
                img = env.render()  # 'rgb_array' mode
            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )

        ac: np.ndarray = policy.get_action(ob)  # ac의 shape은 env.action_space.shape
        next_ob, rwd, done, *_ = env.step(ac)

        steps += 1
        rollout_done = True if (steps == max_traj_length) or done else False

        obs.append(ob)
        acs.append(ac)
        rwds.append(rwd)
        next_obs.append(next_ob)
        terms.append(rollout_done)

        ob = next_ob

        if rollout_done: break

    traj = {
        'obs': np.array(obs, dtype=np.float32),
        'image_obs': np.array(image_obs, dtype=np.uint8),
        'rwds': np.array(rwds, dtype=np.float32),
        'acs': np.array(acs, dtype=np.float32),
        'next_obs': np.array(next_obs, dtype=np.float32),
        'terms': np.array(terms, dtype=np.float32),
    }

    return traj


def sample_trajectories(
        env: gym.Env, policy: nn.Module, min_timesteps_per_batch: int, max_traj_length: int, render: bool = False
) -> Tuple[Trajs, int]:
    """
    총 timestep의 수가 min_timesteps_per_bnatch가 current policy를 env에 굴려 rollouts를 sampling한다.
    ---
    Args:
        min_timesteps_per_batch: batch size
    """
    trajs = []

    timesteps_this_batch = 0
    while timesteps_this_batch < min_timesteps_per_batch:
        traj = sample_trajectory(env, policy, max_traj_length, render)
        trajs.append(traj)

        timesteps_this_batch += get_traj_length(traj)
    batch_size = timesteps_this_batch

    return trajs, batch_size


def sample_n_trajectories(
        env: gym.Env, policy: nn.Module, ntraj: int, max_traj_length: int, render: bool = False
) -> Trajs:
    """
    Current policy를 env에 굴려 ntraj개의 rollouts를 sampling한다.
    """
    trajs = []

    for _ in range(ntraj):
        traj = sample_trajectory(env, policy, max_traj_length, render)
        trajs.append(traj)

    return trajs


def convert_listofrollouts(trajs: Trajs, concat_rwds=True) -> TrajInfo:
    """
    Rollouts가 담긴 list를 각 component별로 concat하고 flatten하여 반환한다.
    """
    observations = np.concatenate([traj['obs'] for traj in trajs])
    actions = np.concatenate([traj['acs'] for traj in trajs])
    if concat_rwds:
        rewards = np.concatenate([traj['rwds'] for traj in trajs])
    else:
        rewards = [traj['rwds'] for traj in trajs]  # List[np.ndarray]
    next_observations = np.concatenate([traj['next_obs'] for traj in trajs])
    terminals = np.concatenate([traj['terms'] for traj in trajs])

    traj_info = (
        observations, actions, rewards, next_observations, terminals
    )

    return traj_info


def get_traj_length(traj: Dict[str, np.ndarray]) -> int:
    return len(traj['rwds'])


def compute_metrics(trajs, eval_trajs):
    """
    Compute (scalar) metrics for logging.
    """
    # return
    train_returns = [traj['rwds'].sum() for traj in trajs]
    eval_returns = [eval_traj['rwds'].sum() for eval_traj in eval_trajs]

    # episode length
    train_epi_lens = [len(traj['rwds']) for traj in trajs]
    eval_epi_lens = [len(eval_traj['rwds']) for eval_traj in eval_trajs]

    logs = OrderedDict()

    logs['Eval_AverageReturn'] = np.mean(eval_returns)
    logs['Eval_StdReturn'] = np.std(eval_returns)
    logs['Eval_MaxReturn'] = np.max(eval_returns)
    logs['Eval_MinReturn'] = np.min(eval_returns)
    logs['Eval_AverageEpLen'] = np.mean(train_epi_lens)

    logs['Train_AverageReturn'] = np.mean(train_returns)
    logs['Train_StdReturn'] = np.std(train_returns)
    logs['Train_MaxReturn'] = np.max(train_returns)
    logs['Train_MinReturn'] = np.min(train_returns)
    logs['Train_AverageEpLen'] = np.mean(eval_epi_lens)

    return logs
