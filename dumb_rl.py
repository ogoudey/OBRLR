import numpy as np
import robosuite as suite

import random
from collections import defaultdict
from itertools import product
import tqdm

import matplotlib.pyplot as plt


class Robot:
    def __init__(self):
        #self.actions = [list(t) for t in product([-1, 0, 1], repeat=7)]
        self.qtable = defaultdict(lambda : { tuple(action): 0.0 for action in [list(t) for t in product([-0.1, 0, 0.1], repeat=7)]})
        self.average_returns = defaultdict(lambda : { tuple(action): [] for action in [list(t) for t in product([-0.1, 0, 0.1], repeat=7)]})
    def best_action(self, obs_vector):
        potential_actions = self.qtable[obs_vector]
        best_actions, best_value = [], -99999
        for action in potential_actions:
            if potential_actions[action] > best_value:
                best_actions = [action]
                best_value = potential_actions[action]
            elif potential_actions[action] == best_value:
                best_actions.append(action)
        return random.choice(best_actions)       

    def policy(self, obs_vector, epsilon=0.1):
        #action = np.random.randn(*env.action_spec[0].shape) * 0.1
        action = self.best_action(obs_vector)
        return action

def main():
# create environment instance
    env = suite.make(
        env_name="Lift", # try with othrrrrrer tasks like "Stack" and "Door"
        robots="Kinova3",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    r = Robot()
    gamma = 0.99


    env.reset()
    obs, __, __, __ = env.step([0,0,0,0,0,0,0])
    obs_vector = (round(obs['robot0_eef_pos'][0], 1), round(obs['robot0_eef_pos'][1], 1), round(obs['robot0_eef_pos'][2], 1)) # override

    num_episodes = 1
    final_rewards = []
    for episode in tqdm.tqdm(range(0, num_episodes)):
        states, actions, rewards = [], [], []
        for i in range(1000):
            states.append(obs_vector)
            action = r.policy(obs_vector)
            actions.append(action)
            #action = [float(input('action[0]')), 0, 0, 0, 0, 0, 0]
            #print(action)
            obs, reward, done, info = env.step(action)  # take action in the environment
            reward = -(np.linalg.norm(obs['robot0_eef_pos'] - obs['cube_pos']))
            rewards.append(reward)
            obs_vector = (round(obs['robot0_eef_pos'][0], 1), round(obs['robot0_eef_pos'][1], 1), round(obs['robot0_eef_pos'][2], 1))


           
            #print(obs, reward, done, info)

            if done:
                for retrospec in range(0, len(states)):
                    _return = 0.0
                    for i in range(retrospec, len(states)):
                        _return += rewards[retrospec]*(gamma**(i - retrospec))
                        
                    r.average_returns[states[retrospec]][actions[retrospec]].append(_return)
                    r.qtable[states[retrospec]][actions[retrospec]] = np.average(r.average_returns[states[retrospec]][actions[retrospec]])
                final_rewards.append(reward)    
                break


        env.reset()
           
    plt.plot(range(0, num_episodes), final_rewards)
    plt.title('final rewards over training in episodes')
    plt.show()
    
    print(r.qtable[states[retrospec]][actions[retrospec]]
    return r

def visualize(r):
    env = suite.make(
        env_name="Lift", # try with othrrrrrer tasks like "Stack" and "Door"
        robots="Kinova3",
        has_renderer=True, # the only difference
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    
    env.reset()
    obs, __, __, __ = env.step([0,0,0,0,0,0,0])
    obs_vector = (round(obs['robot0_eef_pos'][0], 1), round(obs['robot0_eef_pos'][1], 1), round(obs['robot0_eef_pos'][2], 1)) # override
    
    for i in range(0, 1000):
        action = r.policy(obs_vector) 
        obs, reward, done, info = env.step(action)  # reward not used
        print(i, -(np.linalg.norm(obs['robot0_eef_pos'] - obs['cube_pos'])))
        obs_vector = (round(obs['robot0_eef_pos'][0], 1), round(obs['robot0_eef_pos'][1], 1), round(obs['robot0_eef_pos'][2], 1))
        env.render()  # render on display
        if done:
            break



















if __name__ == "__main__":
    r = main()
    visualize(r)
