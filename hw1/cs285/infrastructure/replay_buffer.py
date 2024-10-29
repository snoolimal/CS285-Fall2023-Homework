from cs285.infrastructure.utils import *

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


class ReplayBuffer(object):
    def __init__(self, max_size=1000000):
        self.max_size = max_size

        # 각 rollout을 저장할 그릇
        self.trajs = []

        # 각 rollout의 component arrays를 저장할 그릇
        self.obs = None
        self.acs = None
        self.rwds = None
        self.next_obs = None
        self.terminals = None

    def __len__(self):
        if self.obs:
            return self.obs.shape[0]
        else:
            return 0

    def add_rollouts(self, trajs, concat_rwd=True):
        """
        Buffer가 일정 용량(max_size)를 초과할 경우 오래된 rollouts를 지우고 최신 rollouts를 유지한다.
        ---
        Args:
            concat_rwd: rwd도 array로 concatenate(flatten) vs. (nested) list로 유지
        """
        # rollout paths list에 새로운 rollout paths 추가
        for traj in trajs:
            self.trajs.append(traj)

        # 새로운 rollouts를 componenet arrays로 변환하고 각각을 기존 arrays에 추가
        obs, acs, rwds, next_obs, terminals = (
            convert_list_of_rollouts(trajs, concat_rwd)
        )

        # buffer가 비어 있는 경우
        if self.obs is None:
            # 각 component array로부터 max_size만큼의 최신 experience를 slicing
            self.obs = obs[-self.max_size:]
            self.acs = acs[-self.max_size:]
            self.rwds = rwds[-self.max_size:]
            self.next_obs = next_obs[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        # buffer가 비어 있지 않은 경우
        else:
            # 기존의 experience와 새로운 experience를 이어 붙이고 최신 experience를 max_size만큼 slicing
            self.obs = np.concatenate([self.obs, obs])[-self.max_size:]
            self.acs = np.concatenate([self.acs, acs])[-self.max_size:]
            if concat_rwd:
                self.rwds = np.concatenate([self.rwds, rwds])
            else:
                self.rwds += rwds if isinstance(rwds, list) else self.rwds.append(rwds)
            self.rwds = self.rwds[-self.max_size:]
            self.next_obs = np.concatenate([self.next_obs, next_obs])[-self.max_size:]
            self.terminals = np.concatenate([self.terminals, terminals])[-self.max_size:]
