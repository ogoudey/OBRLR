import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os

class SimWrapper(gym.Env):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self._terminated = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self._terminated = False
        obs = self.sim.observe()
        return obs, {}

    def step(self, action):
        if self._terminated:
            obs, _ = self.reset()
            return obs, 0.0, True, False, {}

        _, _, terminated, _ = self.sim.act(action)
        obs = self.sim.observe()
        reward = self.sim.reward().item()
        #terminated = reward == 1.0
        truncated = reward <= -1.0

        if terminated or truncated:
            self._terminated = True

        return obs, reward, terminated, truncated, {}

    def render(self):
        self.sim.env.render()

def train_ppo(sim, params, args):
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
    model.save(params.get("policy_save_name", "ppo_policy"))
    return model

def test_ppo(sim, model_path="ppo_policy.zip"):
    env = DummyVecEnv([lambda: SimWrapper(sim)])
    model = PPO.load(model_path)

    obs = env.reset()
    while True:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()