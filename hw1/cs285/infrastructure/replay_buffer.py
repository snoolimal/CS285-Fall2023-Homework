from cs285.infrastructure.utils import *


class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.max_size = max_size

        # store each rollout
        self.trajs = []

        # store concatenated component arrays from each rollout
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.terminals = None

    def add_rollouts(self, trajs: Trajs, concat_rwds=True):
        # add new rollouts into our list of rollouts
        for traj in trajs:
            self.trajs.append(traj)

        # convert new rollouts into their component arrays, and append them onto our arrays
        observations, actions, rewards, next_observations, terminals = convert_list_of_rollouts(trajs, concat_rwds)

        # buffer가 비었다면, i.e., ReplayBuffer instance가 생성된 후 첫 add_rollouts() 호출이라면
        # freshest rollouts를 max_size만큼 slicing
        if self.observations is None:
            self.observations = observations[-self.max_size:]
            self.actions = actions[-self.max_size:]
            self.rewards = rewards[-self.max_size:]
            self.next_observations = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        # buffer가 안 비었다면
        # fresh rollouts를 이어 붙이고 freshest rollouts를 max_size만큼 slicing
        else:
            self.observations = np.concatenate([self.observations, observations])[-self.max_size:]
            self.actions = np.concatenate([self.actions, actions])[-self.max_size:]
            if concat_rwds:
                self.rewards = np.concatenate([self.rewards, rewards])[-self.max_size:]
            else:
                self.rewards += rewards
            self.next_observations = np.concatenate([self.next_observations, next_observations])[-self.max_size:]
            self.terminals = np.concatenate([self.terminals, terminals])[-self.max_size:]

    def __len__(self):
        if self.observations is not None:
            return self.observations.shape[0]
        else:
            return 0