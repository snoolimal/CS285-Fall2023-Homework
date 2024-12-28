from abc import ABC, abstractmethod
import numpy as np


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, ob: np.ndarray) -> np.ndarray:
        """
        Sampling single action from conditional distribution fixed by given (single) observation.
        ---
        Args:
            ob: observation | [env.observation_space.shape]
        ---
        Returns:
            ac: action | [env.action_space.shape]
        """
        pass

    @abstractmethod
    def update(self, obs: np.ndarray, acs: np.ndarray, **kwargs) -> dict:
        """
        Batch 단위로 policy를 학습하고 그 과정을 dict로 반환해 logging에 사용한다.
        ---
        Args:
            obs: batch observation [batch size,ob.shape]
            acs: batch action
        """
        pass

    @abstractmethod
    def save(self, filepath: str):
        pass