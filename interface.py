import numpy as np
import robosuite as suite
from robosuite.utils import transform_utils
#from vision import camera_utils as cu
from vision import sim_vision
import torch
from PIL import Image

from scipy.spatial.transform import Rotation as R
import math
import matplotlib.pyplot as plt
import time
import random
from collections import defaultdict
from itertools import product
import tqdm
import json

import copy
import matplotlib.pyplot as plt
import networkx as nx
import torch
import cv2
import os


class Sim:
    def __init__(self, params):
    
        reward_params = params["reward_function"]
        self.env, self.mem_reward, self.state, self.done = None, None, None, 0
        self.eef_pos, self.cube_pos = None, None
        # Initialize (reset)
        self.has_renderer = False
        
        # Vision
        self.sim_vision = sim_vision.SimVision(use_sim_camera=False) # Change to turn simulated YOLO
        
        #
        
        # FOR RAISE REWARD #
        
        #
        has_renderer = False
        use_sim_camera = False
        if "human" in params.keys():
            if bool(params["human"]):
                use_sim_camera = True
                has_renderer = True
        
        if "testing" in params.keys():
            if bool(params["testing"]):
                has_renderer = True
                
        
        
        self.reset(has_renderer=has_renderer, use_sim_camera=use_sim_camera)
        self.initial_cube = self.eef_pos
        
        
        self.use_cost = bool(reward_params["cost"])
        if self.use_cost:
            self.cost_scale = reward_params["cost_scale"]
        self.reward_for_raise = reward_params["raise"]
        
        
        
          
    def reset(self, has_renderer=False, use_sim_camera=False):

        if not self.env:
            print("Full reset!")
            if use_sim_camera:
                self.env = suite.make(
                    env_name="Lift",
                    robots="Kinova3",
                    has_renderer=True,
                    horizon = 1000000,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    camera_heights=400,
                    camera_widths=400,
                    camera_names="sideview",
                    camera_depths=True
                )
            else:
                self.env = suite.make(
                    env_name="Lift",
                    robots="Kinova3",
                    has_renderer=has_renderer,
                    horizon = 1000000,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                )
       
        
        # Set starting joints (a bad starting position after all?)
        desired_joint_positions = [math.pi, 0.0, 0.0, -math.pi/2, 0.0, -math.pi/2, 0]
        self.env.robots[0].set_robot_joint_positions(desired_joint_positions)
        
        
        # Take initial step to get obs
        obs, _, _, _ = self.env.step([0,0,0,0,0,0,0])
        # send obs, etc. to sim_vision for detection. kinda sloppy, we get positions in multiple places
        self.initial_cube = obs['cube_pos']
        detection = self.sim_vision.detect(obs, self.env, no_cap = not has_renderer)
        self.initial_goal = np.array([self.initial_cube[0], self.initial_cube[1], self.initial_cube[2] + 0.05])
        goal = torch.tensor(self.initial_goal, dtype=torch.float32)
        self.cube_pos = detection[3:6]
        self.state = torch.cat((detection, goal))
        
        site_name = self.env.robots[0].gripper['right'].important_sites["grip_site"]
        eef_site_id = self.env.sim.model.site_name2id(site_name)
        self.eef_pos = self.env.sim.data.site_xpos[eef_site_id]
        
        self.sim_vision.reset()
        
        # FOR CUBE Z (for reward)    


        
        self.mem_reward = torch.tensor(self.raise_reward(self.eef_pos, self.initial_cube, self.initial_goal), dtype=torch.float32)
        
    def observe(self):
        state = self.state
        normalized_state = state # i.e. not normalized
        return normalized_state

    def act(self, action, w_video=False):
        standardized_action = [action[0], action[1], action[2], 0, 0, 0, action[3]]
        #print("Standardized action:", standardized_action)
        obs, _, _, _ = self.env.step(standardized_action)
        
        site_name = self.env.robots[0].gripper['right'].important_sites["grip_site"]
        eef_site_id = self.env.sim.model.site_name2id(site_name)
        self.eef_pos = self.env.sim.data.site_xpos[eef_site_id]

        detection = self.sim_vision.detect(obs, self.env, no_cap = not w_video)
        goal = torch.tensor(self.goal(), dtype=torch.float32)
        self.cube_pos = copy.deepcopy(detection[3:6])
        self.state = torch.cat((detection, goal))

        
        ### Update reward ###
        raise_reward = self.calculate_reward(self.eef_pos, self.cube_pos, torch.tensor(self.initial_goal, dtype=torch.float32), standardized_action)
        # stored for access with Sim.reward()
        self.mem_reward = raise_reward
        
        ## Done conditions ##
        if raise_reward == 1:
            self.done = 1
        else:
            self.done = 0      
    
    def calculate_reward(self, eef_pos, cube_pos, cube_goal, action):
        raise_reward = torch.tensor(self.raise_reward(eef_pos, cube_pos, cube_goal), dtype=torch.float32)
        # cost (if applicable)
        if not raise_reward == self.reward_for_raise:
            if self.use_cost:
                raise_reward -= self.torq_cost(action)
        return raise_reward
    
    def goal(self):
        initial = copy.deepcopy(self.initial_cube)
        goal = [initial[0], initial[1], initial[2] + 0.05]
        return np.array(goal)
        
        
    def reward(self):
        return self.mem_reward
    
    def torq_cost(self, action):
        action_norm = np.linalg.norm(action)
        cost = self.cost_scale * action_norm
        #print("Action cost:", cost)
        
        return cost
        
    def raise_reward(self, eef_pos, cube_pos, goal):
        threshold = 0.02
        condition_threshold = 0.05
        #print("Reward calculation:", cube_pos[2], "-", self.initial_cube_z, "=", z_diff, "> 0.01? (1 or 0.1) AND...")
        #print("\teef_pos - cube_pos == X:", eef_pos, "-", cube_pos, "=", np.linalg.norm((eef_pos - cube_pos)), "< 0.04?")
        condition = np.linalg.norm((cube_pos - eef_pos)) < condition_threshold
        delta = np.linalg.norm((goal - cube_pos))
        if delta < threshold and condition:
            return self.reward_for_raise
        else:
            return -0.1
        
    def take_photo(self):
        # For gathering data
        i = random.random()*1000
        obs, _, _, _ = self.env.step([0,0,0,0,0,0,0])
        img = Image.fromarray(obs["sideview_image"], 'RGB')
        img_name = 'sideview'+str(i)+'.png'
        #print("Photo-taking turned off.")
        img.save('vision/data/Robosuite3/Images' + img_name)

# helper function for taking sim photos
def take_onboard_photo(obs):
    i = random.random()*1000
    eye = obs['robot0_eye_in_hand_image']
    img = Image.fromarray(eye, 'RGB')    
    img_name = 'sideview'+str(i)+'.png'    
    img.save('vision/data/Robosuite4/Images' + img_name)    

# main takes sim photos and does teleop to expose things
if __name__ == "__main__":
    env = suite.make(env_name="Lift",
                robots="Kinova3",
                has_renderer=True,
                horizon = 2000,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                camera_heights=256,
                camera_widths=256,
                camera_names="robot0_eye_in_hand",
                camera_depths=False)

    robot = env.robots[0]
    
    
    
    
    obs, _, _, _ = env.step([0,0,0,0,0,0,0])
    print(obs.keys())
    state = robot._hand_pos['right']
    
    speed = 5
    k = 1
    for i in range(0, 4000):
        take_onboard_photo(obs)
        action = np.array([(random.random()-0.5)/k,(random.random()-0.5)/k,(random.random()-0.5)/k,0.0,0.0,0.0,(random.random()-0.5)/10])
        obs, _, _, _ = env.step(action)
        if i % 500 == 0:
            env.reset()
        
        
    while True:
        action = np.array([0.0,0.0,0.0,0.0])
        trigger = input("Button: ")
        if trigger == "q":
            action[0] = speed
        elif trigger == "w":
            action[1] = speed
        elif trigger == "e":
            action[2] = speed
        elif trigger == "r":
            action[3] = speed
        elif trigger == "p":
            state = take_photo()
            break
        else:
            print("Assigning speed!")
            try:
                speed = float(trigger)
            except ValueError:
                print("OOPS!")
                continue
        standardized_action = [action[0], action[1], action[2], 0, 0, 0, action[3]]
        obs, _, _, _ = env.step(standardized_action)

        state = robot._hand_pos['right']
   
        
    # reset sim
    # take 4000 photoes
    
