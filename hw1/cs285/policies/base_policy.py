import abc
import numpy as np


class BasePolicy(metaclass=abc.ABCMeta):
    def get_action(self, ob: np.ndarray) -> np.ndarray:
        """
        Single env-agent interaction을 수행한다.
        ---
        Args:
            ob: curr tstep의 observation vector [ob_dim,]
        Returns:
            ac: action vector [ac_dim,]
        """
        raise NotImplementedError

    def update(self, obs: np.ndarray, acs: np.ndarray, **kwargs) -> dict:
        """
        (Batch 단위로) Policy를 학습하고 그 과정을 반환한다.
        ---
        Args:
            obs: batch obs
            acs: batch acs
        Returns: dict
            logging information
        """
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError