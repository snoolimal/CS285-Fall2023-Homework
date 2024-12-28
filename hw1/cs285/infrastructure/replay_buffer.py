from cs285.infrastructure.utils import *


class ReplayBuffer:
    def __init__(self, max_size: int = 1000000):
        self.max_size = max_size

        self.trajs = []
        self.obs = None
        self.acs = None
        self.rwds = None
        self.next_obs = None
        self.terms = None

    def add_rollouts(self, trajs: Trajs, concat_rwds=True):
        for traj in trajs:
            self.trajs.append(traj)

        # FIFO
        obs, acs, rwds, next_obs, terms = convert_listofrollouts(trajs, concat_rwds)

        # buffer가 비어 있다면
        if self.obs is None:
            # 각 component의 (flatten) array로부터 max_size만큼의 최신 experience를 slicing
            self.obs = obs[-self.max_size:]
            self.acs = acs[-self.max_size:]
            self.rwds = rwds[-self.max_size:]
            self.next_obs = next_obs[-self.max_size:]
            self.terms = terms[-self.max_size:]
        # buffer가 비어 있지 않다면
        else:
            # 기존의 experience와 새로운 experience를 이어 붙이고 최신 experience를 max_size만큼 slicing
            self.obs = np.concatenate([self.obs, obs])[-self.max_size:]
            self.acs = np.concatenate([self.acs, acs])[-self.max_size:]
            if concat_rwds:
                self.rwds = np.concatenate([self.rwds, rwds])[-self.max_size:]
            else:
                self.rwds += rwds
            self.next_obs = np.concatenate([self.next_obs, next_obs])[-self.max_size:]
            self.terms = np.concatenate([self.terms, terms])[-self.max_size:]

    def __len__(self):
        if self.obs is not None:
            return self.obs.shape[0]
        else:
            return 0