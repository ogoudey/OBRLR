import numpy as np
import robosuite as suite

import random
from collections import defaultdict
from itertools import product
import tqdm

import matplotlib.pyplot as plt
import networkx as nx

class Robot:
    def __init__(self):
        #self.actions = [list(t) for t in product([-1, 0, 1], repeat=7)]
        self.qtable = defaultdict(lambda : { tuple(action): 0.0 for action in [list(t) for t in product([-0.1, 0, 0.1], repeat=6)]})
        self.average_returns = defaultdict(lambda : { tuple(action): [] for action in [list(t) for t in product([-0.1, 0, 0.1], repeat=6)]})
        
        # aesthetic
        self.visited_states = set()
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

    num_episodes = 10
    cum_rewards = []
    new_visits_over_episodes = []
    visits_per_episode = []
    closest_distances = []
    
    G = nx.Graph()
    
    num_grades = 10
    grades = dict()
    for num in range(0, num_grades):
        grades[num] = False
    
    episode_length = 500
    for episode in tqdm.tqdm(range(0, num_episodes)):
        initial_distance = np.linalg.norm(obs['robot0_eef_pos'] - obs['cube_pos'])
        
        for grade in grades.keys():
            grades[grade] = False
        cum_reward = 0
        new_visits = 0
        closest_distance = 999
        per_episode_new_visits = 0
        visits = set()
        states, actions, rewards = [], [], []
        for i in range(0, episode_length):
            if obs_vector not in r.visited_states:
                new_visits += 1
                r.visited_states.add(obs_vector)
            if obs_vector not in visits:
                visits.add(obs_vector)
                per_episode_new_visits += 1
            states.append(obs_vector)
            
            G.add_nodes_from([obs_vector])
            action = r.policy(obs_vector)
            actions.append(action)
            #action = [float(input('action[0]')), 0, 0, 0, 0, 0, 0]
            #print(action)
            action = list(action)
            action.append(0.0) # lets ignore the gripper so far
            obs, reward, done, info = env.step(action)  # take action in the environment
            reward = __reward(initial_distance, grades, obs['robot0_eef_pos'], obs['cube_pos'])
            if reward:
                print(reward)
            if i == episode_length - 1 and not done:
                done = True
            rewards.append(reward)
            cum_reward += reward
            obs_vector = (round(obs['robot0_eef_pos'][0], 1), round(obs['robot0_eef_pos'][1], 1), round(obs['robot0_eef_pos'][2], 1))
            closest_distance = min(closest_distance, np.linalg.norm(obs['robot0_eef_pos'] - obs['cube_pos']))
               
           
            #print(obs, reward, done, info)

            if done:
            
                for retrospec in range(0, len(states)):
                    _return = 0.0
                    for i in range(retrospec, len(states)):
                        _return += rewards[retrospec]*(gamma**(i - retrospec))
                        
                    r.average_returns[states[retrospec]][actions[retrospec]].append(_return)
                    r.qtable[states[retrospec]][actions[retrospec]] = np.average(r.average_returns[states[retrospec]][actions[retrospec]])
                cum_rewards.append(cum_reward)
                new_visits_over_episodes.append(new_visits)
                visits_per_episode.append( per_episode_new_visits)
                closest_distances.append(closest_distance)
                break


        env.reset()
        obs, __, __, __ = env.step([0,0,0,0,0,0,0])
        obs_vector = (round(obs['robot0_eef_pos'][0], 1), round(obs['robot0_eef_pos'][1], 1), round(obs['robot0_eef_pos'][2], 1)) # override
        
    fig, ax1 = plt.subplots()       
    plt.plot(range(0, num_episodes), cum_rewards, color='green')
    ax1.set_ylabel('cumulative reward', color='green')
    ax2 = ax1.twinx()
    ax2.plot(range(0, num_episodes), new_visits_over_episodes, color='red')
    ax2.set_ylabel('# new visits', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax3 = ax1.twinx()
    ax3.plot(range(0, num_episodes), visits_per_episode, color='blue')
    ax3.set_ylabel('visits', color='blue')
    ax4 = ax1.twinx()
    ax4.plot(range(0, num_episodes), closest_distances, color='yellow')
    ax4.set_ylabel('closest distance', color='yellow')
    ax4.tick_params(axis='y', labelcolor='yellow')
    plt.title('cumulative rewards, new visits, visited states, closest')
    plt.show()
    
    print(r.qtable[states[retrospec]][actions[retrospec]])
    return r


# Rewards based on how close the eef is to the cube, by grades
# So if there are 10 grades, moving from 1m away to .79m away gives 1 reward and 2 reward at approprate times, for passing .9m and .8m respectively.
def __reward(initial_distance, grades, eef_pos, cube_pos):
    distance = np.linalg.norm(eef_pos - cube_pos)

    for grade in grades.keys():
        if initial_distance / len(grades.keys()) * grade > distance and not grades[grade]:
            grades[grade] = True
            print("Passed", initial_distance / len(grades.keys()) * grade, "with", distance, "Reward:", len(grades.keys()) - grade)
            return len(grades.keys()) - grade
    return 0

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
        action = list(action)
        action.append(0.0) # lets ignore the gripper so far
        obs, reward, done, info = env.step(action)  # reward not used
        
        obs_vector = (round(obs['robot0_eef_pos'][0], 1), round(obs['robot0_eef_pos'][1], 1), round(obs['robot0_eef_pos'][2], 1))
        env.render()  # render on display
        if done:
            break


if __name__ == "__main__":
    r = main()
    visualize(r)
