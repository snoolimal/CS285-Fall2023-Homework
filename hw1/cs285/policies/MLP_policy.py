from __future__ import annotations

from pathlib import Path
import itertools
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.distributions import MultivariateNormal

from cs285.policies.base_policy import BasePolicy
from cs285.infrastructure import pytorch_util as ptu


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        hidden_size: int
) -> nn.Module:
    layers = []
    for _ in range(n_layers):
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Tanh())
        input_size = hidden_size
    layers.append(nn.Linear(hidden_size, output_size))

    mlp = nn.Sequential(*layers)

    return mlp


class MLPPolicySL(BasePolicy, nn.Module):
    """
    Defines an MLP for supervised learning which maps observations to continuous actions,
    which represents the (multivariate) gaussian policy.

    Attributes:
        mean_net: nn.Sequential
            A neural network that outputs the mean of gaussian for continuous actions of each input
        logstd: nn.Parameter
            A separate parameter to learn the standard deviation of actions
            Variance 관련 parameter는 모든 inputs의 gaussians가 공통적으로 사용하도록 설정한다.
            이때 full covariance matrix가 아닌 diagonal matrix를 parameter로 삼아,
            action space의 각 차원이 독립임을 가정한다, i.e., 상관 구조를 modeling하지 않는다.

    Methods:
        forward:
            Runs a differentiable forwards pass through the network
        update:
            Trains the policy with a supervised learning objective
    """
    def __init__(
            self,
            ac_dim: int,
            ob_dim: int,
            n_layers: int,
            hidden_size: int,
            learning_rate: float = 1e-4,
            training: bool = True,
            **kwargs
    ):
        super(MLPPolicySL, self).__init__(**kwargs)

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.training = training

        self.mean_net = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers,
            hidden_size=self.hidden_size
        )
        self.mean_net.to(ptu.device)

        self.logstd = nn.Parameter(
            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )

        self.optimizer = optim.Adam(
            params=itertools.chain(self.mean_net.parameters(), [self.logstd]),
            lr=self.learning_rate
        )

        self.criterion = nn.MSELoss()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            observations: torch.Tensor [B,ob_dim]
                Batch of (concatenated) observations to query the policy

        Returns:
            actions: torch.Tensor [B,ac_dim]
                (Batch of) Sampled actions from the policy (for each given observation)
        """
        mean = self.mean_net(observations)  # B개 observations 각각에 대한 gaussian의 mean vector  [B,ac_dim]
        std = torch.exp(self.logstd)        # 그러한 gaussians가 공유하는 covariance matrix의 diagonal 성분    [ac_dim,]
        policy = MultivariateNormal(mean, scale_tril=torch.diag(std))
        actions = policy.rsample()  # reparameterization

        return actions
        # --- rsample()
        # action을 곧바로 sampling하는 대신 noise를 sampling한다:
        # epsilon = torch.randn_like(mean)
        # action = mean + std * epsilon
        # 그렇다면 sample의 computation graph와의 연결이 유지되므로 gradient가 mean_net의 parameter와 logstd까지 잘 흐른다.
        # ---
        # --- policy = Normal(mean, std)
        # B*ac_dim개의 univariate gaussians가 생성된다.
        # ---

    def update(self, observations: np.ndarray, actions: np.ndarray, **kwargs) -> dict:
        """
        Trains the policy.

        Args:
            observations: np.ndarray [B,ob_dim]
                Batch of (concatenated) observations to query the policy
            actions: np.ndarray [B,ac_dim]
                (Batch of) Actions that we want policy to imiate

        Returns: dict
            Training performance to log
        """
        observations, actions_expert = ptu.from_numpy(observations), ptu.from_numpy(actions)

        self.optimizer.zero_grad()
        actions_pred = self.forward(observations)
        loss = self.criterion(actions_pred, actions_expert)
        loss.backward()
        self.optimizer.step()

        return {
            'Training Loss': loss.item(),
        }

    def get_action(self, ob: np.ndarray) -> np.ndarray:
        """
        Runs the policy for single timestep.

        Args:
            ob: np.ndarray [ob_dim,]
                A given observation from the environment

        Returns:
            ac: np.ndarray [ac_dim,]
                A sampled action from the policy
        """
        assert ob.ndim == 1, ('Need [ob_dim,] for single agent-environment interaction.'
                              'Policy cannt take batch of observations as input while running.')
        ac = self.forward(ptu.from_numpy(ob).unsqueeze(0))  # convert to batch form

        return ptu.to_numpy(ac.squeeze())   # convert from batch form
        # ---
        # 기술적으로 torch.nn으로 쌓은 network는 batch 차원이 없는 single data여도 문제 없이 forwarding하지만,
        # get_action()과 forward()의 호출 상황을 구분하기 위해:
        #   - get_action(): running policy, i.e., get samples
        #   - forward(): training policy using samples
        # 차원을 조작하는 작업을 명시적으로 포함했다.
        # ---

    def save(self, file_path: str | Path):
        torch.save(self.state_dict(), file_path)