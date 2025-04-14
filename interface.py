import numpy as np
import robosuite as suite
from robosuite.utils import transform_utils
from vision import camera_utils as cu
import torch
from PIL import Image

import time
import random
from collections import defaultdict
from itertools import product
import tqdm

import matplotlib.pyplot as plt
import networkx as nx
import torch
import cv2
import os

class Sim:
    def __init__(self):
        self.env, self.reward, self.state = None, None, None
        # Initialize (reset)
        self.has_renderer = False
        
        #FOR SLOPPY REWARDS
        self.num_grades = 10
        self.grades = dict()
        # END FOR SLOPPY REWARDS
        
        self.reset()
        
        
        
          
    def reset(self):
        env = suite.make(
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
        self.env = env
        
        # Initialize
        obs, _, _, _ = self.env.step([0,0,0,0,0,0,0])
        self.reward = torch.tensor(self.__reward(obs['robot0_eef_pos'], obs['cube_pos']), dtype=torch.float32)
        self.state = torch.tensor(np.concatenate((obs['robot0_eef_pos'], obs['cube_pos'])), dtype=torch.float32) # YOLO
        
        # FOR SLOPPY REWARDS
        self.initial_distance = np.linalg.norm(obs['robot0_eef_pos'] - obs['cube_pos'])
        for num in range(0, self.num_grades):
            self.grades[num] = False
        
    def observe(self):
        state = self.state
        return state

    def act(self, action):
        obs, _, _, _ = self.env.step(action)
        reward = torch.tensor(self.__reward(obs['robot0_eef_pos'], obs['cube_pos']), dtype=torch.float32)
        self.reward = torch.tensor([reward]) # ENGINEER
        
        # below should be replaced by YOLO
        state = torch.tensor(np.concatenate((obs['robot0_eef_pos'], obs['cube_pos'])), dtype=torch.float32)
        self.state = state
        
    # SLOPPY REWARD FUNCTION
    def __reward(self, eef_pos, cube_pos):
        
    
        distance = np.linalg.norm(eef_pos - cube_pos)

        for grade in self.grades.keys():
            if self.initial_distance / len(self.grades.keys()) * grade > distance and not self.grades[grade]:
                self.grades[grade] = True
                #prnt("Passed", self.initial_distance / len(grades.keys()) * grade, "with", distance, "Reward:", len(grades.keys()) - grade)
                return (len(self.grades.keys()) - grade) * 1
        return -1
