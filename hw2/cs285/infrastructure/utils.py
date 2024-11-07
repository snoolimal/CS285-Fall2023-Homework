from typing import Tuple, List, Dict
from numpy.typing import NDArray
import numpy as np
import cv2

import gymnasium as gym

from cs285.networks.policies import MLPPolicy


def sample_trajectory(
        env: gym.Env, policy: MLPPolicy, max_length: int, render: bool = False
) -> Dict[str, np.ndarray]:
    """
    Current policy를 env에 굴려 rollout 1개를 sampling한다.
    """
    obs, acs, rwds, next_obs, terms, image_obs = [], [], [], [], [], []
    steps = 0
    ob, _ = env.reset()
    while True:
        if render:
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render()  # 'rgb_array' mode
            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )

        ac: np.ndarray = policy.get_action(ob)
        next_ob, rwd, done, *_ = env.step(ac)

        steps += 1
        rollout_done = True if (steps == max_length) or done else False

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
        env: gym.Env, policy: MLPPolicy, min_timesteps_per_batch: int, max_length: int, render: bool = False
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """
    총 timestep의 수가 min_timesteps_ber_batch가 될 때까지 current policy를 env에 굴려 rollouts를 sampling한다.
    즉, min_timesteps_ber_batch는 batch size이다.
    """
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)

        timesteps_this_batch += get_traj_length(traj)

    return trajs, timesteps_this_batch


def sample_n_trajectories(
    env: gym.Env, policy: MLPPolicy, ntraj: int, max_length: int, render: bool = False
) -> List[Dict[str, np.ndarray]]:
    """
    Current policy를 env에 굴려 ntraj개의 rollouts를 sampling한다.
    """
    trajs = []
    for _ in range(ntraj):
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)

    return trajs


def convert_listofrollouts(
    trajs: List[Dict[str, np.ndarray]]
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, List[NDArray]]:
    """
    Rollouts가 담긴 list를 받아 각 component별로 concat하고 flatten하여 tuple에 담아 반환한다.
    """
    observations = np.concatenate([traj['obs'] for traj in trajs])
    actions = np.concatenate([traj['acs'] for traj in trajs])
    next_observations = np.concatenate([traj['next_obs'] for traj in trajs])
    terminals = np.concatenate([traj['terms'] for traj in trajs])
    rewards = np.concatenate([traj['rwds'] for traj in trajs])
    uncc_rwds = [traj['rwds'] for traj in trajs]

    return (
        observations,
        actions,
        next_observations,
        terminals,
        rewards,
        uncc_rwds
    )


def get_traj_length(traj):
    return len(traj['rwds'])
