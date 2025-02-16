from typing import Union

from abc import ABC, abstractmethod

import numpy as np
from pathlib import Path


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, ob: np.ndarray) -> np.ndarray:
        """
        Policy를 env에 굴릴 때, i.e., every tstep마다 호출하여 rollout을 완성해 나갈 때 사용한다.
        Given single observation으로 fix한 conditional distribution인 policy로부터 single action을 sampling한다.
        ---
        Args:
            ob: given single observation    | [env.observation_space.shape]
                env로부터, i.e., 'env.step()'으로 sampling해 준비해 두었다.
        ---
        Returns:
            ac: single action sample        | [env.action_space.shape]
                policy로부터 sampling한다.
        """
        pass

    @abstractmethod
    def update(self, observations: np.ndarray, actions: np.ndarray, **kwargs) -> dict:
        """
        Policy를 training하고 그 과정에서의 metrics를 dict로 반환하여 (후에) logging에 사용한다.
        ---
        Args:
            observations: batch observation     | [N,ob_dim]
            actions: batch action               | [N,ac_dim]

            cf. ob_dim: env.observation_space.shape
                ac_dim: env.action_space.shape
        """
        pass

    @abstractmethod
    def save(self, filepath: Union[str, Path]):
        pass
