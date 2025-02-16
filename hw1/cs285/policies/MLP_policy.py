import itertools
from pathlib import Path
from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal

from cs285.policies.base_policy import BasePolicy
from cs285.infrastructure import pytorch_util as ptu


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        hidden_size: int,
) -> nn.Sequential:
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
    Continous distribution over action conditioned on observation given ob parameterized by theta.
    Batch 단위로 처리한다.
    Continous distribution은 multivariate gaussian으로 상정한다.
    ---
    Attributes:
        mean_net: observation을 mean vector에 mapping하는 neural network (policy를 parameterize)
        logstd: mean과 같이 learnable하게 처리하되
                observation에 의존하는 variable인 mean vector와 달리 constant인 log-standard deviation
    """
    def __init__(
            self,
            ac_dim: int,
            ob_dim: int,
            n_layers: int,
            hidden_size: int,
            lr: float = 1e-4,
            training: bool = True,
            nn_baseline: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.lr = lr
        self.training = training
        self.nn_baseline = nn_baseline

        self.mean_net: nn.Sequential = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers,
            hidden_size=self.hidden_size
        )
        self.mean_net.to(ptu.device)

        self.logstd = nn.Parameter(
            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)

        self.optimizer = optim.Adam(
            params=itertools.chain([self.logstd], self.mean_net.parameters()),
            lr=self.lr
        )

        self.criterion = nn.MSELoss()

    def forward(self, observations: torch.FloatTensor) -> torch.Tensor:
        """
        Args:
            observations: batch observation         | [N,ob_dim]
        ---
        Returns:
            action_pred: batch actions prediction   | [N,ac_dim]
        """
        mean = self.mean_net(observations)  # N개의 training points 각각에 대응하는 mean vector  | [N,ac_dim]
        std = torch.exp(self.logstd)        # N개의 training points에 공통적으로 사용하는 cov max의 diag     | [ac_dim,]
        dist = MultivariateNormal(mean, scale_tril=torch.diag(std))     # 여기까진 문제없이 computation graph가 연결

        actions_pred = dist.rsample()    # reparameterization trick으로 computation graph 연결 유지
        return actions_pred

    def update(self, observations: np.ndarray, actions: np.ndarray, **kwargs) -> dict:
        """
        Args:
            observations: batch observation     | [N,ob_dim]
            actions: batch expert action        | [N,ac_dim]
        ---
        Returns: training performance
        """
        observations = ptu.to_tensor(observations)
        actions_expert = ptu.to_tensor(actions)

        self.optimizer.zero_grad()
        actions_pred = self.forward(observations)
        loss = self.criterion(actions_pred, actions_expert)
        loss.backward()
        self.optimizer.step()

        return {
            'Training Loss': loss.item()
        }

    def get_action(self, ob: np.ndarray) -> np.ndarray:
        """
        Args:
            ob: single observation  | [ob_dim]
        ---
        Returns:
            ac: single action       | [ac_dim]
        """
        ac = self.forward(ptu.to_tensor(ob))
        return ptu.to_numpy(ac)

    def save(self, filepath: Union[str, Path]):
        if isinstance(filepath, Path): filepath = str(filepath)
        torch.save(self.state_dict(), filepath)
