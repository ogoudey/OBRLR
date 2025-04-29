import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os

def train_ppo(sim, params, args):
    class SimWrapper(gym.Env):
        def __init__(self, sim):
            super().__init__()
            self.sim = sim
            obs = self.sim.observe()
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            self._terminated = False  # <-- track termination internally

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.sim.reset()
            self._terminated = False  # reset termination flag
            obs = self.sim.observe()
            info = {}
            return obs, info

        def step(self, action):
            if self._terminated:
                # After episode is done, just return dummy values until reset
                obs = self.sim.observe()  # Or you could return np.zeros_like(self.sim.observe())
                reward = 0.0
                terminated = True
                truncated = False
                info = {}
                return obs, reward, terminated, truncated, info

            self.sim.act(action)
            obs = self.sim.observe()
            reward = self.sim.reward().item()
            terminated = reward == 1.0
            truncated = reward <= -1.0

            if terminated or truncated:
                self._terminated = True

            info = {}
            return obs, reward, terminated, truncated, info

        def render(self, mode='human'):
            pass


    env = DummyVecEnv([lambda: Monitor(SimWrapper(sim))])
    experiment_name = "ppo"
    tensorboard_log_path = os.path.join("logs", experiment_name)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log_path,
        **params.get("ppo_kwargs", {})
    )
    model.learn(total_timesteps=params.get("total_timesteps", 100_000))
    model.save(params.get("policy_save_name", "ppo_bird_sim"))
    return model

def test_ppo(sim, model_path="models/ppo_model.zip"):
    class SimWrapper(gym.Env):
        def __init__(self, sim):
            super().__init__()
            self.sim = sim
            obs = self.sim.observe()
            act = np.zeros_like(obs)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=act.shape, dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.sim.reset()
            obs = self.sim.observe()
            info = {}
            return obs, info

        def step(self, action):
            self.sim.act(action)
            obs = self.sim.observe()
            reward = self.sim.reward().item()
            terminated = reward == 1.0
            truncated = reward <= -1.0
            info = {}
            return obs, reward, terminated, truncated, info 

        def render(self, mode='human'):
            pass

    env = DummyVecEnv([lambda: SimWrapper(sim)])
    model = PPO.load(model_path)

    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
