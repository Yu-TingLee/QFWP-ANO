"""The RL environments and their episode budgets."""

import operator
from functools import reduce

import gymnasium as gym
import minigrid  # noqa: F401  -- registers the MiniGrid-* env ids
import numpy as np

# Episodes per run. Every arm and seed of an env shares its budget, so the
# episode axis is comparable across arms.
EPISODE_BUDGET = {
    "Acrobot-v1": 5000,
    "MiniGrid-SimpleCrossingS9N1-v0": 10000,
}
ENV_IDS = tuple(EPISODE_BUDGET)


class ImgObsFlatWrapper(gym.ObservationWrapper):
    """Keep only the flattened MiniGrid image, dropping the mission string."""

    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space.spaces["image"]
        img_size = reduce(operator.mul, img_space.shape, 1)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(img_size,), dtype=np.float32
        )

    def observation(self, obs):
        return obs["image"].flatten().astype(np.float32)


class Float32Wrapper(gym.ObservationWrapper):
    """Cast observations to float32, which A3C's v_wrap expects."""

    def __init__(self, env):
        super().__init__(env)
        space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=space.low, high=space.high, shape=space.shape, dtype=np.float32
        )

    def observation(self, obs):
        return np.asarray(obs, dtype=np.float32)


def make_env(env_id):
    env = gym.make(env_id)
    if env_id.startswith("MiniGrid"):
        return ImgObsFlatWrapper(env)
    return Float32Wrapper(env)
