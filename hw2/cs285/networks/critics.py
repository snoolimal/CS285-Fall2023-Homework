import numpy as np

import torch
from torch import nn
from torch import optim

from cs285.infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    """Value Network
    Batch observation을 받아 각 ob의 value를 뱉는다.
    """
    def __init__(
            self,
            ob_dim: int,
            n_layers: int,
            hidden_size: int,
            lr: float,
    ):
        super().__init__()

        self.v_phi_pi = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            hidden_size=hidden_size,
        ).to(ptu.device)
        params = self.v_phi_pi.parameters()

        self.optimizer = optim.Adam(
            params, lr
        )

        self.criterion = nn.MSELoss()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim < 2: observations.unsqueeze(dim=0)  # batch 처리를 기본으로
        return self.v_phi_pi(observations).squeeze()

    def update(self, observations: np.ndarray, q_mc_estimates: np.ndarray) -> dict:
        observations = ptu.from_numpy(observations)
        q_mc_estimates = ptu.from_numpy(q_mc_estimates)
        optimizer = self.optimizer
        criterion = self.criterion

        optimizer.zero_grad()
        values = self(observations).squeeze()
        loss = criterion(q_mc_estimates, values)
        loss.backward()
        optimizer.step()

        return {
            'Baseline Loss': loss.item()
        }


if __name__ == "__main__":
    # for debug
    params = {
        'ob_dim': 3, 'n_layers': 2, 'hidden_size': 10, 'lr': 1e-2,
    }
    ntrajs, tsteps = 5, 10
    batch_size = ntrajs * tsteps
    observations = np.random.randn(batch_size, params['ob_dim'])
    mc_estimates = np.random.randn(batch_size)
    V = ValueCritic(**params)
    _ = V.update(observations, mc_estimates)
