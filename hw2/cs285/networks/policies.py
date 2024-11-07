from abc import ABC, abstractmethod
import itertools
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical, MultivariateNormal

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module, ABC):
    """Base MLP Policy
    Batch observation을 받아 distribution over actions를 뱉는다.
    ---
    이 class에서는 `forward`와 `get_action` methods만 구현한다.
    PG algorithm의 종류에 따라 obj grad가 달리 계산되므로, `update` method는 subclass에서 구현한다.
    """
    def __init__(
            self,
            ac_dim: int,
            ob_dim: int,
            discrete: bool,
            n_layers: int,
            hidden_size: int,
            lr: float,
    ):
        super().__init__()

        if discrete:
            # ac_dim=4: 로봇의 팔 1개가 동서남북 중 어디 방향으로?
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                hidden_size=hidden_size,
            ).to(ptu.device)
            params = self.logits_net.parameters()
        else:
            # ac_dim=4: 로봇의 팔다리 4개는 몇 도만큼 굽힐까?
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                hidden_size=hidden_size
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            params = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            params=params, lr=lr
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, ob: np.ndarray) -> np.ndarray:
        """
        Agent-Env loop의 single interaction을 수행한다.
        즉, single observation으로 policy를 query해 single action을 반환한다.
        ---
        Args:
            ob: np.ndarray [ob_dim,]
        Returns:
            ac: np.ndarray [ac_dim,]
        """
        ob = ptu.from_numpy(ob)
        dist = self(ob)
        ac = dist.sample()
        # ac_prob = torch.exp(dist.log_prob(ac))

        return ptu.to_numpy(ac)

    def forward(self, observations: torch.Tensor) -> torch.distributions:
        """
        Policy is a (parameterized) distribution over action space conditioned on the observation.
        """
        if self.discrete:
            logits = self.logits_net(observations)  # parameterized, conditioned on observation
            dist = Categorical(logits=logits)       # distribution over action space
        else:
            mean, std = self.mean_net(observations), torch.exp(self.logstd)
            dist = MultivariateNormal(mean, scale_tril=torch.diag(std))

        return dist

    @abstractmethod
    def update(self, observations: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        Batch 입력을 받아 1 gradient ascent step을 밟는다.
        """
        pass


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the Policy Gradient Algorithm
    PG algorithm의 종류에 따라 obj grad가 달리 계산되므로
    그 다양성을 효율적으로 커버하기 위하여 update method를 다르게 override한 적절한 subclasses를 만들어 사용한다.
    """
    def update(self, observations: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        Policy gradient actor update를 수행한다.
        """
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(kwargs.get('advantage_esstimates'))
        optimizer = self.optimizer

        optimizer.zero_grad()
        dist = self(observations)
        log_probs = dist.log_prob(actions)  # log-lh
        assert log_probs.shape == advantages.shape
        weighted_neg_log_lh = torch.neg(torch.mul(log_probs, advantages))
        pseudo_loss = torch.mean(weighted_neg_log_lh)
        pseudo_loss.backward()
        optimizer.step()

        return {
            'Actor Loss': pseudo_loss.item()
        }


if __name__ == "__main__":
    # for debug
    import gymnasium as gym
    from cs285.infrastructure.utils import *
    env = gym.make('Ant-v5')
    params = {
        'ob_dim': env.observation_space.shape[0],
        'n_layers': 2, 'hidden_size': 10, 'lr': 1e-2,
    }
    if isinstance(env.action_space, gym.spaces.Discrete):   # Discrete vs. Box
        params['discrete'] = True
        params['ac_dim'] = env.action_space.n
    else:
        params['discrete'] = False
        params['ac_dim'] = env.action_space.shape[0]
    policy = MLPPolicyPG(**params)
    ntraj, max_length = 5, 10
    trajs = sample_n_trajectories(env, policy, ntraj, max_length)
    observations, actions, _, terminals, rewards, _ = convert_listofrollouts(trajs)
    advantage_estimates = np.random.randn(len(observations))     # batch size
    info = policy.update(observations, actions, advantage_esstimates=advantage_estimates)
    print('')
