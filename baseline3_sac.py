import gym
import numpy as np
from gym import spaces
from stable_baselines3 import SAC, HER
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

def train_sac(sim, params, args):

    # Goal-conditioned environment wrapper for Sim
    class KinovaLiftEnv(gym.GoalEnv):
        def __init__(self, sim_env, threshold=0.01):
            super().__init__()
            self.sim = sim_env
            self.threshold = threshold
            
            # Get initial observation to define space shapes
            obs = self.sim.observe()
            obs = np.array(obs, dtype=np.float32)
            # Define observation and goal spaces
            obs_dim = obs.shape if isinstance(obs, np.ndarray) else (len(obs),)
            
            
            # Set achieved and desired goal to be 1-dimensional (lift height)
            self.observation_space = spaces.Dict({
                'observation': spaces.Box(low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32),
                'achieved_goal': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                'desired_goal': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
            })
            # Action space: 4D continuous (x, y, z, gripper), each in [-1, 1]
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            # Store success reward value and cost usage from Sim for reward computation
            self.success_reward = self.sim.reward_for_raise if hasattr(self.sim, "reward_for_raise") else 1.0
            self.use_cost = getattr(self.sim, "use_cost", False)
            self.cost_scale = getattr(self.sim, "cost_scale", 1.0)
            
            # Internal step counter for episode length management
            self.max_steps = params.get("algorithm", {}).get("len_episode", 50)
            self.current_step = 0

        # Reset the environment and return initial observation, achieved goal, and desired goal
        def reset(self):
            
            self.sim.reset()
            self.current_step = 0
            obs = np.array(self.sim.observe(), dtype=np.float32)
            # Achieved goal is zero lift at start (cube at initial position)
            achieved = np.array([0.0], dtype=np.float32)
            # Desired goal is the lift threshold (target height)
            desired = np.array([self.threshold], dtype=np.float32)
            return {'observation': obs, 'achieved_goal': achieved, 'desired_goal': desired}

        # Step function to apply action, get new observation, compute reward, and check done condition
        def step(self, action):
            # Clip action to -1 to 1 range
            action = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)
            self.sim.act(action)
            obs = np.array(self.sim.observe(), dtype=np.float32)
            
            
            # Compute achieved goal (current cube lift height) and desired goal
            z_diff = self.sim.cube_pos[2] - self.sim.initial_cube_z
            achieved = np.array([z_diff], dtype=np.float32)
            desired = np.array([self.threshold], dtype=np.float32)
            
            
            # Get reward from sim
            reward = self.sim.reward().item()
            # Determine success condition for info
            success = bool(z_diff > self.threshold and np.linalg.norm(self.sim.eef_pos - self.sim.cube_pos) < 0.04)
            # Episode done if success or large negative reward (failure)
            done = bool(reward >= self.success_reward or reward <= -1.0)
            self.current_step += 1
            
            # Also end episode if max steps reached (time limit)
            if self.current_step >= self.max_steps:
                done = True
            info = {'is_success': success}
            
            # If using cost, compute it from action
            # and add to info (same as sim.torq_cost)
            # HER will use this info for reward computation
            if self.use_cost:
                # Calculate torque cost from action (same as sim.torq_cost)
                cost = self.cost_scale * np.linalg.norm(action)
                info['cost'] = cost
            return ({'observation': obs, 'achieved_goal': achieved, 'desired_goal': desired},
                    reward, done, info)

        # Compute reward based on achieved goal, desired goal, and info
        # This is used by HER to compute the reward for the sampled goals
        def compute_reward(self, achieved_goal, desired_goal, info):
            # achieved_goal and desired_goal are arrays of shape (1,)
            z_diff = achieved_goal[0]
            threshold = desired_goal[0]
            
            # Check success condition (lift > threshold and eef-cube dist < 0.04)
            # If info is provided, use eef-cube distance from there
            # otherwise check the distance
            if info is not None and 'is_success' in info:
                success = bool(info['is_success'])
            else:
                # If no info, estimate success by checking distance
                success = bool(z_diff > threshold)
            if success:
                reward = float(self.success_reward)  
            else:
                reward = -0.1
                # Include torque cost penalty
                if self.use_cost and info is not None and 'cost' in info:
                    reward -= info['cost']
            return reward

    # Instantiate the GoalEnv wrapper
    env = KinovaLiftEnv(sim, threshold=params.get("lift_threshold", 0.01))
    # wrap with TimeLimit to enforce max episode steps
    env = gym.wrappers.TimeLimit(env, max_episode_steps=params.get("algorithm", {}).get("len_episode", 50))

    # Prepare HER + SAC model
    model_class = SAC 
    n_sampled_goal = params.get("n_sampled_goal", 4)  # number of HER replays per episode
    goal_selection = params.get("goal_selection_strategy", "future")  
    online_sampling = params.get("online_sampling", True)
    total_timesteps = params.get("total_timesteps", 100_000)
    model_save_path = params.get("policy_save_name", "sac_kinova_lift_model")

    # Choose the policy type for Dict observation space
    policy_type = "MultiInputPolicy"  

    # Initialize HER with SAC model
    model = HER(
        policy_type,
        env,
        model_class,
        n_sampled_goal=n_sampled_goal,
        goal_selection_strategy=goal_selection,
        online_sampling=online_sampling,
        max_episode_length=params.get("algorithm", {}).get("len_episode", 50),
        verbose=1,
        **params.get("sac_kwargs", {})
    )
    # Train the agent
    model.learn(total_timesteps=total_timesteps)
    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved as {model_save_path}")
    return model
