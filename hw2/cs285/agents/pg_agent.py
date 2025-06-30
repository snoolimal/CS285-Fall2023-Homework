from typing import Optional, List
from numpy.typing import NDArray
import numpy as np

from torch import nn

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu


class PGAgent(nn.Module):
    def __init__(
            self,
            ac_dim: int,
            ob_dim: int,
            discrete: bool,
            n_layers: int,
            hidden_size: int,
            lr: float,
            gamma: float,
            use_basline: bool,
            baseline_lr: Optional[float],
            baseline_grad_steps: Optional[int],
            gae_lambda: Optional[float],
            use_reward_to_go: bool,
            normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, hidden_size, lr
        )

        # create the critic (baseline) network, if needed
        if use_basline:
            self.critic = ValueCritic(
                ob_dim, n_layers, hidden_size, baseline_lr
            )   # actor network와 n_layers와 hidden size가 같을 필욘 없지만 편의를 위해
            self.baseline_grad_steps = baseline_grad_steps
            self.gae_lambda = gae_lambda
        else:
            self.critic = None

        # other agent params
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.normalize_advantages = normalize_advantages

    def update(
            self,
            observations: List[np.ndarray],
            actions: List[np.ndarray],
            rewards: List[np.ndarray],
            terminals: List[np.ndarray],
    ) -> dict:
        """
        Train step은 아래의 작동으로 구성된다:
        ---
        각 입력은 ndarray의 list이며, 각 list는 single traj에 대응한다.
        그러므로 batch size는 모든 trajs의 tsteps의 총합이다.
        """
        q_mc_estimates = self._calculate_q_mc_estimate(rewards)

        observations = np.concatenate(observations)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        terminals = np.concatenate(terminals)
        q_mc_estimates = np.concatenate(q_mc_estimates)

        advantage_estimates: np.ndarray = self._estimate_advantage(
            q_mc_estimates, observations, rewards, terminals
        )

        info: dict = self.actor.update(observations, actions, advantage_esstimates=advantage_estimates)
        
        if self.critic is not None:
            for _ in range(self.baseline_grad_steps):
                critic_info: dict = self.critic.update(observations, q_mc_estimates)

            info.update(critic_info)

        return info

    def _estimate_advantage(
            self,
            q_mc_estimates: np.ndarray,
            observations: np.ndarray,
            rewards: np.ndarray,
            terminals: np.ndarray,
    ):
        if self.critic is None:
            advantage_estimates = q_mc_estimates
        else:
            values = ptu.to_numpy(self.critic(ptu.from_numpy(observations)))
            assert values.shape == q_mc_estimates.shape

            if self.gae_lambda is None:
                advantage_estimates = q_mc_estimates - values
            else:
                batch_size = observations.shape[0]

                values = np.append(values, [0])
                advantage_estimates = np.zeros(batch_size + 1)
                for t in reversed(range(batch_size)):
                    delta_t = rewards[t] + ((1 - terminals[t]) * self.gamma * values[t + 1]) - values[t]
                    advantage_estimate_t = delta_t + self.gamma * self.gae_lambda * advantage_estimates[t + 1]
                    advantage_estimates[t] = advantage_estimate_t
                advantage_estimates = advantage_estimates[:-1]

        if self.normalize_advantages:
            advantages = (advantage_estimates - np.mean(advantage_estimates)) / (np.std(advantage_estimates))

        return advantages

    def _calculate_q_mc_estimate(self, rewards: List[np.ndarray]) -> List[np.ndarray]:
        """Monte Carlo Estimation of the Q Function"""
        if not self.use_reward_to_go:
            q_mc_estimates = [self._discounted_return(rwds) for rwds in rewards]
        else:
            q_mc_estimates = [self._discounted_reward_to_go(rwds) for rwds in rewards]

        return q_mc_estimates

    def _discounted_reward_to_go(self, rwds: NDArray[np.floating]) -> NDArray[np.floating]:
        dc_rwds_to_go = []
        dc_rwd_to_go = 0.
        for r in reversed(rwds):
            dc_rwd_to_go = r + (self.gamma * dc_rwd_to_go)
            dc_rwds_to_go.append(dc_rwd_to_go)
        dc_rwds_to_go = np.array(dc_rwds_to_go)[::-1]

        return dc_rwds_to_go

    def _discounted_return(self, rwds: NDArray[np.floating]) -> NDArray[np.floating]:
        ds_factors = np.power(self.gamma, np.arange(len(rwds)))
        dc_returns = np.full(rwds.shape, np.sum(ds_factors * rwds))
        return dc_returns


if __name__ == "__main__":
    # for debug
    from cs285.infrastructure.utils import *
    env = gym.make('CartPole-v1')    # 'Ant-v5' or 'CartPole-v1'
    params = {
        'ob_dim': env.observation_space.shape[0],
        'n_layers': 2, 'hidden_size': 5, 'lr': 1e-2, 'gamma': 0.9,
        'use_basline': True, 'baseline_lr': 1e-3, 'baseline_grad_steps': 5,
        'gae_lambda': 0.9,
        'use_reward_to_go': True, 'normalize_advantages': True,
    }
    if isinstance(env.action_space, gym.spaces.Discrete):  # Discrete vs. Box
        params['discrete'] = True
        params['ac_dim'] = env.action_space.n
    else:
        params['discrete'] = False
        params['ac_dim'] = env.action_space.shape[0]

    pg_agent = PGAgent(**params)

    ntraj, max_length = 5, 10
    observations, actions, terminals, rewards = [], [], [], []
    for n in range(ntraj):
        traj = sample_trajectory(env, pg_agent.actor, max_length)
        observations.append(traj['obs'])
        actions.append(traj['acs'])
        terminals.append(traj['terms'])
        rewards.append(traj['rwds'])

    info = pg_agent.update(observations, actions, rewards, terminals)
    print(info)
