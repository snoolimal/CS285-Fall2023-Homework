from typing import Union

import itertools
import numpy as np
from pathlib import Path

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
    def __init__(
            self,
            ac_dim: int,
            ob_dim: int,
            n_layers: int,
            hidden_size: int,
            lr: float = 1e-4,
            training: bool = True,
            # nn_baseline: bool = False,
    ):
        super().__init__()

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.lr = lr
        self.training = training
        # self.nn_baseline = nn_baseline

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

    def update(self, observations: np.ndarray, actions: np.ndarray, **kwargs) -> dict:
        """
        Train the policy.
        ---
        Args:
            observations: batch observation     | [N,ob_dim]
            actions: batch action               | [N,ac_dim]
            **kwargs:
        ---
        Returns: training performance(s) (metric(s) to log)
        """
        observations = ptu.to_tensor(observations)
        actions_expert = ptu.to_tensor(actions)

        self.optimizer.zero_grad()
        actions = self.forward(observations)
        loss = self.criterion(actions, actions_expert)
        loss.backward()
        self.optimizer.step()

        return {
            'Training Loss': loss.item()
        }

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass the batch component(s).

        Policy는 특정 tstep의 observation에서 취할 action을 선택한다.
        i.e., given condition o_t로 policy distribution을 fix하고 a_t를 sampling한다.
        get_action()과 forward()는 모두 위 작동을 구현하지만, 아래와 같이 다른 쓰임을 갖는다:
            - get_action()
                Policy를 env에 굴리는 rollout sampling에 사용한다.
                Rollout은 (components의) sequential한 sampling으로 완성되어야 한다, i.e., time dependency가 존재한다.
                그러므로 single 단위의 sampling이 총 timestep의 for loop에서 반복적으로 이루어져야 한다.
            - forward()
                Forward pass 후 얻은 action sample로 loss를 계산하고 back propagate하는 policy training에 사용한다.
                Training에서는 temporal한 action sampling이 아닌
                input(given) observation에 대한 최종 output으로써의 action sample, i.e., 각 observation에 대한 action의
                mapping을 다루며, 그러한 mapping으로 하여금 human expert의 그것을 clone하도록 만드는 것이 목적이다.
                그러므로 여기서는 action sampling에 있어 time dependency를 고려할 필요 없이,
                여러 tsteps의 observation을 한번에 -- 굳이 연속적일 필요도 없다 -- 받고
                (각 observation point에 대한) action을 sampling하는 batch 단위의 처리를 해도 문제가 없다.
        이렇듯 쓰임은 다르지만, 구조적으로는 get_action()은 forward()의 N=1의 case이다.
        ---
        Args:
            observations: batch observation             | [N,ob_dim]
        ---
        Returns:
            actions_pred = batch action parediction     | [N,ac_dim]
        """
        mean = self.mean_net(observations)  # [N,ac_dim], N개의 training points에 대응하는 (policy dist의) mean vector
        std = torch.exp(self.logstd)        # [ac_dim,], N개의 training points에 공통적으로 사용하는 각 차원의 std
        dist = MultivariateNormal(mean, scale_tril=torch.diag(std))     # 여기까지는 문제 없이 computation graph가 연결
                                                                        # alt. Normal(mean, std)
        actions = dist.rsample()    # reparameterization trick으로 sample에도 computation graph의 연결 유지
        return actions

    def get_action(self, ob: np.ndarray) -> np.ndarray:
        """
        Args:
            ob: single observation  | [ob_dim,]
        ---
        Returns:
            ac: single action       | [ac_dim,]
        """
        # ac = self.forward(ptu.to_tensor(ob))
        # return ptu.to_numpy(ac)
        ac = self.forward(ptu.to_tensor(ob).unsqueeze(0))  # batch form으로
        return ptu.to_numpy(ac.squeeze())
        # ---
        # ```
        # ac = self.forward(ptu.to_tensor(ob))
        # return ptu.to_numpy(ac)
        # ```
        # 위와 같이 batch 차원을 명시적으로 추가하지 않아도 build_mlp()로 만든 policy network는 자동으로 차원 처리를 수행해 준다.
        # 근데 헷갈리니까 batch 차원을 그냥 명시적으로 나타내자.
        # ---

    def save(self, filepath: Union[str, Path]):
        torch.save(self.state_dict(), str(filepath))
