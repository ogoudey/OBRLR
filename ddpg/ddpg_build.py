# ddpg.py
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

def make_ddpg(env_fn, cfg):
    """
    Build a SB3 DDPG model using the given env factory and config dict.
    """
    # vectorize (single env) so SB3 always sees a VecEnv
    vec_env = DummyVecEnv([env_fn])

    # create OU action noise
    n_actions = vec_env.action_space.shape[-1]
    sigma = cfg["training"].get("action_noise_sigma", 0.1)
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=sigma * np.ones(n_actions),
    )

    model = DDPG(
        "MlpPolicy",
        vec_env,
        action_noise=action_noise,
        learning_rate=cfg["training"]["learning_rate"],
        buffer_size=cfg["training"]["buffer_size"],
        batch_size=cfg["training"]["batch_size"],
        gamma=cfg["training"]["gamma"],
        tau=cfg["training"]["tau"],
        verbose=1,
        tensorboard_log=cfg["logging"]["tensorboard_log"],
    )
    return model
