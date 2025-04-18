import numpy as np
import robosuite as suite
from robosuite.utils import transform_utils
#from vision import camera_utils as cu
from vision import sim_vision
import torch
from PIL import Image

import matplotlib.pyplot as plt
import time
import random
from collections import defaultdict
from itertools import product
import tqdm
import json

import matplotlib.pyplot as plt
import networkx as nx
import torch
import cv2
import os


class Sim:
    def __init__(self):
        self.env, self.mem_reward, self.state = None, None, None
        # Initialize (reset)
        self.has_renderer = False
        
        # Vision
        self.sim_vision = sim_vision.SimVision()
        
        #
        
        # FOR RAISE REWARD #
        self.initial_cube_z = 0
        #
        
        #FOR SLOPPY REWARDS
        self.num_grades = 10
        self.grades = dict()
        self.initial_distance = 0
        # END FOR SLOPPY REWARDS
        
        self.reset()
        
        
        
          
    def reset(self, has_renderer=False):
        if not self.env:
            self.env = suite.make(
                env_name="Lift",
                robots="Kinova3",
                has_renderer=self.has_renderer,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                horizon = 1000,
                camera_heights=400,
                camera_widths=400,
                camera_names="sideview",
                camera_depths=True,
            )
        else:
            self.env.has_renderer=has_renderer
            self.has_renderer = has_renderer
            self.env.reset()

        
        # Initialize
        obs, _, _, _ = self.env.step([0,0,0,0,0,0,0])
        self.mem_reward = torch.tensor(self.raise_reward(obs['cube_pos']), dtype=torch.float32)
        self.state = self.sim_vision.detect(obs["sideview_image"], obs["sideview_depth"], self.env.sim)
        
        self.sim_vision.reset()
        
        # FOR CUBE Z        
        self.initial_cube_z = obs['cube_pos'][2]
        
        # FOR SLOPPY REWARDS
        self.initial_distance = np.linalg.norm(obs['robot0_eef_pos'] - obs['cube_pos'])
        for num in range(0, self.num_grades):
            self.grades[num] = False
        
    def observe(self):
        state = self.state
        normalized_state = state # i.e. not normalized
        return normalized_state

    def act(self, action):
        obs, _, _, _ = self.env.step(action)
        self.mem_reward = torch.tensor(self.raise_reward(obs['cube_pos']), dtype=torch.float32)
        self.state = self.sim_vision.detect(obs["sideview_image"], obs["sideview_depth"], self.env.sim, no_cap= not self.has_renderer)
        
        
        
    def reward(self):
        return self.mem_reward
        
    def raise_reward(self, cube_pos):
        diff = (cube_pos[2] - self.initial_cube_z)
        print("Reward calculation:", cube_pos[2], "-", self.initial_cube_z, "=", diff)
        
        if diff > .01:
            return 1
        else:
            return -0.1
            
    
    # SLOPPY REWARD FUNCTION
    def make_reward(self, eef_pos, cube_pos):
        distance = np.linalg.norm(eef_pos - cube_pos)

        for grade in self.grades.keys():
            if self.initial_distance / len(self.grades.keys()) * grade > distance and not self.grades[grade]:
                self.grades[grade] = True
                #prnt("Passed", self.initial_distance / len(grades.keys()) * grade, "with", distance, "Reward:", len(grades.keys()) - grade)
                return (len(self.grades.keys()) - grade) * 1
        return -1


if __name__ == "__main__":
    s = Sim()
    s.has_renderer = True
    s.reset()
    while True:
        action = np.array(json.loads(input("Action: ")))
        obs, _, _, _ = s.env.step(action)
        reward = torch.tensor(s.raise_reward(obs['cube_pos']), dtype=torch.float32)
        print(torch.tensor([reward]))
        state = torch.tensor(np.concatenate((obs['robot0_eef_pos'], obs['cube_pos'])), dtype=torch.float32)
        print(state)
    
