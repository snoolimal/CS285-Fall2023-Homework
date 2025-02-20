from typing import Union

from abc import ABC, abstractmethod

import numpy as np
from pathlib import Path


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, ob: np.ndarray) -> np.ndarray:
        """
        Policy, i.e., conditional distribution를
            1. given single observation으로 fix하고
            2. single action을 sampling한다.
        ---
        Args:
            ob: given single observation   | [env.observation_space.shape,]
        ---
        Returns:
            ac: single action sample        | [env.action_space.shape,]
        """
        pass

    @abstractmethod
    def update(self, observations: np.ndarray, actions: np.ndarray, **kwargs) -> dict:
        """
        Train the policy.
        ---
        Args:
            observations: batch observation     | [N,env.observation_spahce.shape]
            actions: batch action               | [N,env.action_space.shape]
            **kwargs:
        ---
        Returns: training performance(s) (metric(s) to log)
        """
        pass

    @abstractmethod
    def save(self, filepath: Union[str, Path]):
        """
        Save the policy.
        """
        pass