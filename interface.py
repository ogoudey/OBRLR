import numpy as np
import logging

# Grab the robosuite logger
logger = logging.getLogger("robosuite")

# 1) Set it to only allow CRITICAL
logger.setLevel(logging.CRITICAL)

# 2) Remove any handlers that might already be attached
for h in list(logger.handlers):
    logger.removeHandler(h)

# 3) Prevent any records from bubbling up to root
logger.propagate = False
import warnings
warnings.filterwarnings("ignore")
import robosuite as suite
from robosuite.utils import transform_utils
from robosuite.utils.placement_samplers import UniformRandomSampler
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
    def __init__(self, show=False):
    
        self.mem_reward, self.state, self.done = None, None, 0
        self.eef_pos, self.initial_cube, self.cube_pos, self.initial_goal = None, None, None, None
        self.obs = 0
        # Initialize (reset)

        
        # Vision
        self.sim_vision = sim_vision.SimVision(use_sim_camera=False) # Change to turn simulated YOLO
        
        self.env = suite.make(
            env_name="Lift",
            robots="Kinova3",
            has_renderer=show,
            horizon = 1000000,
            has_offscreen_renderer=False,
            use_camera_obs=False,
        )
        
        """
        For copying purposes
        

        env = suite.make(
            env_name="Lift",
            robots="Kinova3",
            has_renderer=True,
            horizon = 1000000,
            has_offscreen_renderer=False,
            use_camera_obs=False,
        )
        
        """
        
    def compose(self, composition, inter_test_memory=None):
        # first, if composition should be skipped
        
        
        self.env.sim.data.qvel[:] = 0.0
        self.env.sim.data.qacc[:] = 0.0
        
        
        # Cube and eef
        if "reset_eef_randomize_cube" == composition:
            desired_joint_positions = [0.0, math.pi/4, 0.0, math.pi/2, 0.0, math.pi/4, -math.pi/2]
            self.env.robots[0].set_robot_joint_positions(desired_joint_positions)
            self.env.sim.data.set_joint_qpos(self.env.model.mujoco_objects[0].joints[0], np.array([(random.random()-0.5)/4,(random.random()-0.5)/4,0.822,1.0,0.0,0.0,0.0]))
        if "reset_eef" == composition:
            desired_joint_positions = [0.0, math.pi/4, 0.0, math.pi/2, 0.0, math.pi/4, -math.pi/2]
            self.env.robots[0].set_robot_joint_positions(desired_joint_positions)
            self.env.sim.data.set_joint_qpos(self.env.model.mujoco_objects[0].joints[0], np.array([0.0,0.0,0.822,1.0,0.0,0.0,0.0]))
        elif "overall" == composition:
            desired_joint_positions = [0.0, math.pi/4, 0.0, math.pi/2, 0.0, math.pi/4, -math.pi/2]
            self.env.robots[0].set_robot_joint_positions(desired_joint_positions)
            self.env.sim.data.set_joint_qpos(self.env.model.mujoco_objects[0].joints[0], np.array([0.0,0.0,0.822,1.0,0.0,0.0,0.0]))
        elif "midway_eef" == composition:
            if not inter_test_memory:
                desired_joint_positions = [-0.06872584,  1.095,  0.0,  1.43028395,  0.00620798,  0.57827479, -math.pi/2]  
                self.env.robots[0].set_robot_joint_positions(desired_joint_positions)
                self.env.sim.data.set_joint_qpos(self.env.model.mujoco_objects[0].joints[0], np.array([0.0,0.0,0.822,1.0,0.0,0.0,0.0]))
        if inter_test_memory:
            desired_joint_positions = inter_test_memory["eef_joints"]
            self.env.robots[0].set_robot_joint_positions(desired_joint_positions)   
            self.env.sim.data.set_joint_qpos(self.env.model.mujoco_objects[0].joints[0], np.concatenate((inter_test_memory["cube_pos"], np.array([1.0,0.0,0.0,0.0]))))
            
        # Gripper
        if "reset_eef" == composition:
            self.set_gripper(-1.0)
        elif "reset_eef_randomize_cube" == composition:
            self.set_gripper(-1.0)
        elif "overall" == composition:
            self.set_gripper(-1.0)
        elif "midway_eef" == composition:
            self.set_gripper(1.0)
        elif "just_grip" == composition:
            self.set_gripper(1.0)

        # Take initial step to get obs for elsewhere and for initial_cube
        self.obs, _, _, _ = self.env.step([0,0,0,0,0,0,0])
        
        self.initial_cube = self.obs['cube_pos']
         
        if "overall" == composition:
            self.initial_eef_goal = copy.deepcopy(self.initial_cube)
            self.initial_cube_goal = np.array([self.initial_cube[0], self.initial_cube[1], self.initial_cube[2] + 0.05])
        elif "midway_eef" == composition:
            self.initial_cube_goal = np.array([self.initial_cube[0], self.initial_cube[1], self.initial_cube[2] + 0.05]) # The actual goal
    
    def epilogue(self, epilogue):
        # Only called during testing
        print("Epilogue: dropping...")
        if epilogue == "drop":
            self.set_gripper(-1.0)
            time.sleep(3)
        
    def set_gripper(self, amount):
        if amount > 0:
            for i in range(0, 10):
                self.env.step([0,0,0,0,0,0,1])
        if amount < 0:
            for i in range(0, 10):
                self.env.step([0,0,0,0,0,0,-1]) 
        """
        robot = env.robots[0]
        target = robot.gripper['right'].format_action(np.array([amount]))
        for name, q in zip(robot.gripper['right'].joints, target):
            jid = env.sim.model.joint_name2id(name)
            env.sim.data.qpos[env.sim.model.jnt_qposadr[jid]] = q
        env.sim.forward()
        """
        
        
        #self.sim_vision.reset() # only needed for YOLO
        
        #self.mem_reward = torch.tensor(self.raise_reward(self.eef_pos, self.initial_cube, self.initial_goal), dtype=torch.float32) # shouldn't need this
    def eef_cube_distance(self, scale):
        displacement = self.eef_pos - self.cube_pos # or initial_cube_goal
        distance = np.linalg.norm(displacement)
        reward = -1 * distance * scale
        return reward
        
    def cube_cube_distance(self, scale):
        displacement = self.initial_cube_goal - self.cube_pos
        distance = np.linalg.norm(displacement)
        reward = -1 * distance * scale
        return reward

    def eef_cube_displacement(self):
        # semantic level
        return self.eef_pos - self.cube_pos
        
    def cube_cube_displacement(self):
        # semantic level - this and the above are the same
        return self.initial_cube_goal - self.cube_pos
    
    def k_cube_eef_distance(self, k):
        reward = 0
        if np.linalg.norm(self.initial_cube_goal - self.eef_pos) < k:
            reward += 1
            self.done = 1
        return reward
        
    def k_cube_cube_distance(self, k):
        reward = 0
        #print("Cube->Cube Displacement:", self.initial_goal - self.cube_pos)
        if np.linalg.norm(self.initial_cube_goal - self.cube_pos) < k:
            reward += 1
            self.done = 1
        return reward
            
    def get_eef_pos(self):
        site_name = self.env.robots[0].gripper['right'].important_sites["grip_site"]
        eef_site_id = self.env.sim.model.site_name2id(site_name)
        self.eef_pos = self.env.sim.data.site_xpos[eef_site_id]
        return self.eef_pos

    def get_cube_pos(self):
        detection = self.sim_vision.detect(self.obs, w_video=False)  
        self.cube_pos = detection
        return self.cube_pos


    
    def get_current_grasp(self):
        gripper_pos = self.obs["robot0_gripper_qpos"]  # CHANGE!!  (?)
        x = gripper_pos.mean()
        opening = np.tanh((x - 0.03367756) * 10)
        
        return np.array([opening])

        
    def observe(self):
        state = self.state
        normalized_state = state # i.e. not normalized
        return normalized_state

    def act(self, standardized_action, w_video=False):

        #print("Standardized action:", standardized_action)
        self.obs, _, _, _ = self.env.step(standardized_action)   

    
    def get_initial_cube_goal(self):
        return self.initial_cube_goal
    
    def calculate_reward(self, eef_pos, cube_pos, cube_goal, action):
        raise_reward = torch.tensor(self.raise_reward(eef_pos, cube_pos, cube_goal), dtype=torch.float32)
        # cost (if applicable)
        if not raise_reward == self.reward_for_raise:
            if self.use_cost:
                raise_reward -= self.torq_cost(action)
        return raise_reward
    
    
        
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
        
        condition = np.linalg.norm((cube_pos - eef_pos)) < condition_threshold
        delta = np.linalg.norm((goal - cube_pos))
        if delta < threshold and condition:
            return self.reward_for_raise
        else:
            return 0.0
        
    def take_photo(self):
        # For gathering data
        i = random.random()*1000
        obs, _, _, _ = self.env.step([0,0,0,0,0,0,0])
        img = Image.fromarray(obs["sideview_image"], 'RGB')
        img_name = 'sideview'+str(i)+'.png'
        #print("Photo-taking turned off.")
        img.save('vision/data/Robosuite3/Images' + img_name)

    def get_inter_test_memory(self):
        d = dict()
        d["cube_pos"] = self.get_cube_pos()
        d["eef_joints"] = self.env.robots[0].recent_qpos.current

        return d
        
    def close(self):
        self.env.close()
        
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
    env.robots[0].control.joint_pos
    speed = 5
    k = 1
    
        
    # Should we ever reset the environment?
    
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
            num_photos = int(input("# photos to take: "))
            for i in tqdm.tqdm(range(0, num_photos)):
                take_onboard_photo(obs)
                action = np.array([(random.random()-0.5)/k,(random.random()-0.5)/k,(random.random()-0.5)/k,0.0,0.0,0.0,(random.random()-0.5)/10])
                obs, _, _, _ = env.step(action)
        elif trigger == "c":
            env.reset()       
                
                #state = take_photo()
                    
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
    
