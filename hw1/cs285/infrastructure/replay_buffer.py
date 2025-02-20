from cs285.infrastructure.utils import *


class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.max_size = max_size

        # store each rollout
        self.trajs = []

        # store component arrays from each converted rollout
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.terminals = None

    def add_rollouts(self, trajs: Trajs, concat_rwds: bool = True):
        """
        최대 용량을 넘어서면 오래된 rollout의 components를 지우고 최신 rollout의 그것을 유지한다.
        """
        # add new rollouts into list of rollouts
        for traj in trajs:
            self.trajs.append(traj)

            # convert new rollouts into their component arrays, and append them onto
            observations, actions, rewards, next_observations, terminals = convert_list_of_rollouts(trajs, concat_rwds)

            # buffer가 완전히 비어 있다면
            if self.observations is None:
                # 각 component array로부터 max_size만큼의 최신 experience를 slicing
                self.observations = observations[-self.max_size:]
                self.actions = actions[-self.max_size:]
                self.rewards = rewards[-self.max_size:]
                self.next_observations = next_observations[-self.max_size:]
                self.terminals = terminals[-self.max_size:]
            # buffer가 완전히 비어 있지 않다면
            else:
                # 기존의 experience와 새로운 experience를 이어 붙이고 최신 experience를 max_size만큼 slicing
                self.observations = np.concatenate([self.observations, observations])[-self.max_size:]
                self.actions = np.concatenate([self.actions, actions])[-self.max_size:]
                if concat_rwds:
                    self.rewards = np.concatenate([self.rewards, rewards])
                else:
                    self.rewards += rewards if isinstance(rewards, list) else np.append(self.rewards, rewards)
                self.rewards = self.rewards[-self.max_size:]
                self.next_observations = np.concatenate([self.next_observations, next_observations])[-self.max_size:]
                self.terminals = np.concatenate([self.terminals, terminals])[-self.max_size:]

    def __len__(self):
        if self.observations is not None:
            return self.observations.shape[0]
        else:
            return 0