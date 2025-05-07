# interface.py
import numpy as np
import gym
from gym import spaces

class RobosuiteGymWrapper(gym.Env):
    """
    Wraps a robosuite env to the standard Gym API,
    flattening state-based OrderedDict observations to a 1D array.
    """
    def __init__(self, env):
        super().__init__()
        self.env = env
        
        # Gym action space from robosuite's action_spec
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Gym observation space: infer from a single reset()
        obs_dict = self.env.reset()
        flat = self._flatten_obs(obs_dict)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=flat.shape, dtype=np.float32
        )

    def _flatten_obs(self, obs_dict):
        
        return np.concatenate([obs_dict[k].ravel() for k in sorted(obs_dict.keys())], axis=0)

    def reset(self):
        obs = self.env.reset()
        return self._flatten_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._flatten_obs(obs), reward, done, info

    def render(self, mode="human"):
        return self.env.render(mode=mode)
