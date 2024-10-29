"""
Some miscellaneous utility functions.
"""

import cv2
import numpy as np
from collections import OrderedDict


"""
길이 T의 single trajectory는 다음과 같은 dict이다:
{'obs': np.ndarray([o1, ..., oT]),
 (optinal) 'image_obs': np.ndarray([io1, ..., ioT]),
 'acs': np.ndarray([a1, ..., aT]),
 'rwds': np.ndarray([r1, ..., oT]),
 'next_obs': np.ndarray([no1, ..., noT]),
 'terminals': np.ndarray([False, False, ..., True]),
}
이떄 observation space와 action space의 dim에 따라 obs, acs, next_obs는 [T,dim]의 array가 된다.
N개의 trajs가 모여 batch가 구성된다.
이러한 batch가 list에 담겨 전체 dataset을 구성한다.

cf. trajectory, rollout, episode를 혼용하며,
    이를 구성하는 env와의 single tstep interaction으로 agent가 받은 값들은 experience라 한다.
"""


class TrajSampler:
    """
    Agent를 env에 굴려 rollout(trajectory)를 sampling한다.
    """
    def __init__(self, env, policy, max_traj_len, render=False):
        """
        Args:
            render: flag of ob의 image rendering
        """
        self.env = env
        self.policy = policy
        self.max_traj_len = max_traj_len
        self.render = render

    def single(self):
        """
        Agent를 env에 굴려 rollout(trajectory)를 "하나" sampling한다.
        ---
        Returns: dict
            각 componenet key에 mapping된 experience arrays
        """
        env, policy, render, max_traj_len = self.env, self.policy, self.render, self.max_traj_len

        # rollout의 experience component 그릇
        obs, acs, rwds, next_obs, terminals, image_obs = [], [], [], [], [], []
        steps = 0

        # 새로운 rollout 시작을 위한 env init
        ob, _ = self.env.reset()  # initial ob after resetting the env

        # sampling single trajectory
        while True:
            if render:
                if hasattr(env, 'sim'):
                    img = env.sim.render(camera_name='track', height=500, width=500)[::-1]
                else:
                    img = env.render(mode='single_rgb_array')
                image_obs.append(cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))

            ac = policy.get_action(ob)
            next_ob, rwd, done, *_ = env.step(ac)

            steps += 1
            rollout_done = True if (done or steps >= max_traj_len) else False

            # experience 기록
            obs.append(ob)
            acs.append(ac)
            rwds.append(rwd)
            next_obs.append(next_ob)
            terminals.append(rollout_done)

            ob = next_ob

            if rollout_done:
                break

        return {
            'obs': np.array(obs, dtype=np.float32),
            'image_obs': np.array(image_obs, dtype=np.uint8),
            'rwds': np.array(rwds, dtype=np.float32),
            'acs': np.array(acs, dtype=np.float32),
            'next_obs': np.array(next_obs, dtype=np.float32),
            'terminals': np.array(terminals, dtype=np.float32)
        }

    def batch_mintstep(self, min_tsteps_per_batch):
        """
        Batch 속 trajs가 밟은 tstep의 총합이 flag값 이상이 되로록 trajs를 sampling한다.
        ---
        Args:
            min_tsteps_per_batch: flag
        Returns: (list, int)
            1. batch (trajs의 list)
            2. batch의 총 tstep
        """
        env, policy, render, max_traj_len = self.env, self.policy, self.render, self.max_traj_len

        total_tstep_this_batch = 0
        trajs = []

        while total_tstep_this_batch < min_tsteps_per_batch:
            traj = TrajSampler(env, policy, max_traj_len, render).single()
            trajs.append(traj)
            total_tstep_this_batch += TrajSampler.get_traj_len(traj)
        # trajs = trajs[0]

        return trajs, total_tstep_this_batch

    def batch_ntraj(self, n_traj):
        """
        n_traj개의 trajs로 구성된 batch를 sampling한다.
        ---
        Returns: list
            batch
        """
        env, policy, render, max_traj_len = self.env, self.policy, self.render, self.max_traj_len

        trajs = []
        for _ in range(n_traj):
            traj = TrajSampler(env, policy, max_traj_len, render).single()
            trajs.append(traj)
        # trajs = trajs[0]

        return trajs

    @staticmethod
    def get_traj_len(traj):
        return len(traj['rwds'])


def convert_list_of_rollouts(trajs, concat_rwd=True):
    """
    Take a list of rollout dicts with list values of component keys
    and return seperate componenet arrays concatenated across the rollouts.
    ---
    Args:
        concat_rwd: rwd도 array로 concatenate(flatten) vs. (nested) list로 유지
    """
    obs = np.concatenate([traj['obs'] for traj in trajs])
    acs = np.concatenate([traj['acs'] for traj in trajs])
    if concat_rwd:
        rwds = np.concatenate([traj['rwds'] for traj in trajs])
    else:
        rwds = [traj['rwds'] for traj in trajs]
    next_obs = np.concatenate([traj['next_obs'] for traj in trajs])
    terminals = np.concatenate([traj['terminals'] for traj in trajs])

    return obs, acs, rwds, next_obs, terminals


def compute_metrics(trajs, eval_trajs):
    """
    Performance를 측정한다.
    return은 logging에 사용한다.
    """
    # return
    train_returns = [traj['rwds'].sum() for traj in trajs]
    eval_returns = [eval_traj['rwds'].sum() for eval_traj in eval_trajs]

    # episode length
    train_ep_lens = [len(traj['rwds']) for traj in trajs]
    eval_ep_lens = [len(eval_traj['rwds']) for eval_traj in eval_trajs]

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
