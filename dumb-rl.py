import numpy as np
import robosuite as suite

import random
from collections import defaultdict
from itertools import product

# create environment instance
env = suite.make(
    env_name="Lift", # try with othrrrrrer tasks like "Stack" and "Door"
    robots="Kinova3",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
)

class Robot:
    def __init__(self):
        self.actions = [list(t) for t in product([-1, 0, 1], repeat=7)]
        self.qtable = defaultdict(lambda : { tuple(action): 0.0 for action in [list(t) for t in product([-0.1, 0, 0.1], repeat=7)]})

    def best_action(self, obs_vector):
        potential_actions = self.qtable[obs_vector]
        best_actions, best_value = [], -99999
        for action in potential_actions:
            if potential_actions[action] > best_value:
                return action
            elif potential_actions[action] == best_value:
                best_actions.append(action)
        return random.choice(best_actions)       

    def policy(self, obs_vector, epsilon=0.1):
        #action = np.random.randn(*env.action_spec[0].shape) * 0.1
        action = self.best_action(obs_vector)
        return action

r = Robot()
gamma = 0.999


env.reset()
obs, __, __, __ = env.step([0,0,0,0,0,0,0])
obs_vector = (round(obs['robot0_eef_pos'][0], 1), round(obs['robot0_eef_pos'][1], 1), round(obs['robot0_eef_pos'][2], 1)) # override
while True:
    states, actions, rewards = [], [], []
    for i in range(1000):
        states.append(obs_vector)
        action = r.policy(obs_vector)
        actions.append(action)
        #action = [float(input('action[0]')), 0, 0, 0, 0, 0, 0]
        #print(action)
        obs, reward, done, info = env.step(action)  # take action in the environment
        reward = np.linalg.norm(obs['robot0_eef_pos'] - obs['cube_pos'])
        rewards.append(reward)
        obs_vector = (round(obs['robot0_eef_pos'][0], 1), round(obs['robot0_eef_pos'][1], 1), round(obs['robot0_eef_pos'][2], 1))


        #
        #
        # Keep track of averages!!
        #
        #

        
        
        #print(obs, reward, done, info)
        env.render()  # render on display
        if done:
            for retrospec in range(0, len(states)):
                for i in range(retrospec, len(states)):
                    r.qtable[states[retrospec]][actions[retrospec]] += reward*(gamma**(i - retrospec))
            break
        print(rewards)
    print(r.qtable)  
    env.reset()
       


