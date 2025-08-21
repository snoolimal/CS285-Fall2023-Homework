from __future__ import annotations

from abc import ABC, abstractmethod

from pathlib import Path
import numpy as np


class BasePolicy(ABC):
    @abstractmethod
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
        pass

    @abstractmethod
    def update(self, observations: np.ndarray, actions: np.ndarray, **kwargs) -> dict:
        """
        Trains the policy.

        Args:
            observations: np.ndarray [B,ob_dim]
                Batch of (concatenated) observations to query the policy
            actions: np.ndarray [B,ac_dim]
                Batch of actions that corresponds to observations

        Returns: dict
            Training performance to log
        """
        pass

    @abstractmethod
    def save(self, file_path: str | Path):
        """
        Saves the trained policy.
        """
        pass