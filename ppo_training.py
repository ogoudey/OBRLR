import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train_ppo(sim, params, args):
    class SimWrapper(gym.Env):
        def __init__(self, sim):
            super().__init__()
            self.sim = sim
            obs = self.sim.observe()
            act = np.zeros_like(obs)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=act.shape, dtype=np.float32)

        def reset(self):
            self.sim.reset()
            return self.sim.observe()

        def step(self, action):
            self.sim.act(action)
            obs = self.sim.observe()
            reward = self.sim.reward().item()
            done = reward == 1.0 or reward <= -1.0
            return obs, reward, done, {}

        def render(self, mode='human'):
            pass

    env = DummyVecEnv([lambda: SimWrapper(sim)])
    model = PPO("MlpPolicy", env, verbose=1, **params.get("ppo_kwargs", {}))
    model.learn(total_timesteps=params.get("total_timesteps", 100_000))
    model.save(params.get("policy_save_name", "ppo_bird_sim"))
    return model

def test(sim, model_path="ppo_bird_sim"):
    class SimWrapper(gym.Env):
        def __init__(self, sim):
            super().__init__()
            self.sim = sim
            obs = self.sim.observe()
            act = np.zeros_like(obs)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=act.shape, dtype=np.float32)

        def reset(self):
            self.sim.reset()
            return self.sim.observe()

        def step(self, action):
            self.sim.act(action)
            obs = self.sim.observe()
            reward = self.sim.reward().item()
            done = reward == 1.0 or reward <= -1.0
            return obs, reward, done, {}

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
