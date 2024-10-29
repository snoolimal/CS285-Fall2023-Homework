"""
Defines a PyTorch policy as the agent's actor.
"""

from typing import Any

import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal

from cs285.policies.base_policy import BasePolicy
from cs285.infrastructure import pytorch_util as ptu


def build_mlp(
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int
) -> nn.Module:
    """
    Builds a feed forward neural network.
    ---
    Args:
        input_size: 입력층의 노드 수 (size of input layer)
        hidden_size: 은닉층의 노드 수 (dimension of each hidden layer)
        output_size: 출력층의 노드 수 (size of output layer)
        n_layers: 은닉층의 수
    Returns:
        MLP (nn.Module)
    """
    layers = []
    in_features = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_features, hidden_size))
        layers.append(nn.Tanh())
        in_features = hidden_size
    layers.append(nn.Linear(in_features, output_size))

    mlp = nn.Sequential(*layers)

    return mlp


class MLPPolicySL(BasePolicy, nn.Module):
    def __init__(
            self,
            ac_dim: int,
            ob_dim: int,
            n_layers: int,
            hidden_size: int,
            learning_rate=1e-4,
            training=True,
            nn_baseline=False,
            multivariate=True
    ):
        super().__init__()

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline
        self.multivariate = multivariate    # annotation: univariate vs. multivariate

        # init learnable params (computation graph에 등록)
        self.mean_net = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers
        ).to(ptu.device)

        if not multivariate:
            # ac_dim개의 univariate gaussian distribution 각각의 logstd
            #   일반화된 관점에서는 multivariate의 cov mat을 diag로 만들어 상관관계를 나타내는 값인 non-diag entry는 모두 0으로 하는 것
            self.entry = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
        else:
            # 입력 space의 차원이 ac_dim인 multivariate gau dist의 cov mat을 만들기 위한 준비 (log_cholesky_elements)
            self.entry = nn.Parameter(
                torch.full(size=(int(ac_dim * (ac_dim + 1) / 2),), fill_value=0.1,
                           dtype=torch.float32, device=ptu.device)
            )

        # set optimizer
        self.optimizer = optim.Adam(
            params=itertools.chain([self.entry],
                                   self.mean_net.parameters()),  # 여러 iterable을 하나의 iterable로 연결
            lr=self.learning_rate
        )

        # set loss func -- i.e., objective func --
        self.criterion = nn.MSELoss()

    def get_action(self, ob: np.ndarray) -> np.ndarray:
        ac = self.forward(ptu.from_numpy(ob))
        return ptu.to_numpy(ac)

    def update(self, obs: np.ndarray, acs: np.ndarray, **kwargs):
        """
        Updates and trains the policy.
        ---
        Args:
            obs: batch의 observations
            acs:
                batch의 actions that we want our policy to imitate -- i.e., batch label --
        Returns:
            performance log: dict
                {'Training Loss': supervised learning loss}
        """
        obs = ptu.from_numpy(obs)
        expert_acs = ptu.from_numpy(acs)

        self.optimizer.zero_grad()
        acs_pred = self.forward(obs)
        loss = self.criterion(acs_pred, expert_acs)
        loss.backward()
        self.optimizer.step()

        return {
            'Training Loss': loss.item()
        }

    def forward(self, obs: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the neural network.
        ---
        Args:
            obs: (batch) obs | [batch_size,traj_len,ob_dim]
        Returns:
            acs: (batch) acs SAMPLED from policy (policy는 distribution) | [batch_size,traj_len,ac_dim]
        ---
        annitation: you can return anything you want, but you should be able to differentiate through it
        """
        mean = self.mean_net(obs)  # 각 given ob vec에 대해 ac의 dist(를 특정하는 param(mean) vec)로 구성된 batch

        # case 1: univariate gaussian policy로 설정
        if not self.multivariate:
            std = torch.exp(self.entry)     # exp(logstd)
            distribution = Normal(mean, std)
        # case 2: multivariate guassian policy로 설정
        else:
            covariance_matrix = self._make_cov_mat(mean)
            distribution = MultivariateNormal(mean, covariance_matrix)

        action = distribution.rsample()     # annotation: reparameterization trick

        return action

    def save(self, filepath):
        """
        Args:
            filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    def _make_cov_mat(self, mean):
        assert self.multivariate

        L = torch.zeros(self.ac_dim, self.ac_dim, dtype=torch.float32, device=ptu.device)
        idx = torch.tril_indices(row=self.ac_dim, col=self.ac_dim, offset=0)
        L[idx[0], idx[1]] = self.entry  # log_cholesky_elements
        L = L + torch.eye(self.ac_dim, dtype=torch.float32, device=mean.device) * 1e-3
        covariance_matrix = L @ L.T

        return covariance_matrix

    @staticmethod
    def check():
        import gymnasium as gym
        import numpy as np
        from cs285.scripts.params import ArgParser

        params = ArgParser.dummy_params_for_debug()

        env = gym.make(params['env_name'])     # 'Ant-v4'
        ac_dim, ob_dim = env.action_space.shape[0], env.observation_space.shape[0]
        n_layers, hidden_size = params['n_layers'], params['hidden_size']
        batch_size = params['batch_size']
        print('action space: {}\n'
              'observation space: {}'
              .format(env.action_space, env.observation_space))
        print('\n'
              'hyperparameters\n'
              'n_layers: {}\n'
              'hidden_size: {}\n'
              'batch_size: {}'.
              format(n_layers, hidden_size, batch_size))
        policy = MLPPolicySL(ac_dim, ob_dim, n_layers, hidden_size)

        batch_obs = np.random.randn(batch_size, ob_dim)     # dummy input(obs)
        batch_acs = np.random.randn(batch_size, ac_dim)     # dummy label(acs)
        print('\n'
              'batch_obs (input): {}\n'
              'batch_acs (label): {}'
              .format(batch_obs.shape, batch_acs.shape))
        policy.update(batch_obs, batch_acs)


if __name__ == "__main__":
    MLPPolicySL.check()
