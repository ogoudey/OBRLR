import numpy as np
import robosuite as suite
from robosuite.utils import transform_utils
import camera_utils as cu
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


class Robot:
    def __init__(self):
        self.vision_model = torch.hub.load('ultralytics/yolov5', 'custom', path='vision2/best.pt', force_reload=True) # path is to the model originally in runs/expX/weights/
        

    def policy(self, obs_vector, epsilon=0.1):
        action = np.random.randn(7) * 0.1
        return action

def make_env(has_renderer=True):
    return suite.make(
        env_name="Lift",
        robots="Kinova3",
        has_renderer=has_renderer,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        horizon = 1000,
        camera_heights=400,
        camera_widths=400,
        camera_names="sideview",
        camera_depths=True,
    )

def detect():
    cap = cv2.VideoCapture(0)      
    env = make_env()

    r = Robot()
    while True:
        obs, reward, done, info = env.step(r.policy(None))  # take action in the environment   
        img = Image.fromarray(obs["sideview_image"], 'RGB')
        result = r.vision_model(img)
        renderings = result.render()
        cv2.imshow("Detections", renderings[0])
        key = cv2.waitKey(30)  
        if key & 0xFF == 27:  # ESC key to break out of the loop
            break

def detect_from_env(env):
    #cap = cv2.VideoCapture(0)      

    r = Robot()

    obs, reward, done, info = env.step(r.policy(None))  # take action in the environment   
    img = Image.fromarray(obs["sideview_image"], 'RGB')
    result = r.vision_model(img)
    return box_centers()



def positions_from_labelled_pixels(depth, env, labelled_pixel_dict):
    object_3d_positions = dict()
    expanded_depth = depth[np.newaxis, ...]

    world_to_pixel_transform = cu.get_camera_transform_matrix(env.sim, "sideview", 400, 400)
    camera_to_world_transform = np.linalg.inv(world_to_pixel_transform)
    for obj in labelled_pixel_dict.keys():
        points = cu.transform_from_pixels_to_world(np.array([labelled_pixel_dict[obj]]), expanded_depth, camera_to_world_transform)
        object_3d_positions[obj] = points[0]
    return object_3d_positions

def box_centers(result):
    centers = dict()
    box_per_obj_cat = defaultdict(lambda : False) # Right now we are gauranteed one box per item. Not necesarily the best box...
    for detection in result.xywh[0]:
        if not box_per_obj_cat[detection[5]]:
            centers[result.names[detection[5].item()]] = (detection[1].item(), detection[2].item())
    

    return centers

def to_state(observation, state):
    print(observation)
    print(state)
    for obj in observation.keys():
        state[obj] = observation[obj]
    for key in state.keys():
        if key not in observation.keys():
            print(key, "did not get updated because it was not seen.")
    return state

def main(imshow=True, has_renderer=True):
    env = make_env(has_renderer=has_renderer)

    r = Robot()
    
    # Learning parameters
    gamma = 0.99
    
    ###
    
    
    # More parameters
    num_episodes = 10 # No. training episodes
    episode_length = 500    
    ###
    
    # Reward structure (see `reward()`)
    num_grades = 10
    grades = dict()
    for num in range(0, num_grades):
        grades[num] = False
    ###

    # Viewing stuff
    if imshow:
        cap = cv2.VideoCapture(0)  
    ###

    print("Starting episodes...")
    for episode in tqdm.tqdm(range(0, num_episodes)):
        
        env.reset() # This call could be improved. We do not need to reset the whole environment, just the positions.
        for grade in grades.keys():
            grades[grade] = False
        
        obs, __, __, __ = env.step([0,0,0,0,0,0,0])
        state = defaultdict(lambda : "UNDETECTED")
        state = to_state(observe(env, r, obs), state)
        
        
        initial_distance = np.linalg.norm(obs['robot0_eef_pos'] - obs['cube_pos'])
        
        for i in range(0, episode_length):
            action = r.policy(state)
            obs, reward, done, info = env.step(action)  # take action in the environment
            state = to_state(observe(env, r, obs), state) # EEF position, cube position
            # KF?
            print("Cube | Computed:", state["cube"], "| Objective:", obs['cube_pos'])
            print("Eef | Computed:", state["eef"], "| Objective:", obs['robot0_eef_pos'])
            
            
            reward = __reward(initial_distance, grades, obs['robot0_eef_pos'], obs['cube_pos']) # remains objective

            if i == episode_length - 1:
                done = True

            obs_vector = (round(obs['robot0_eef_pos'][0], 1), round(obs['robot0_eef_pos'][1], 1), round(obs['robot0_eef_pos'][2], 1))
             
            if has_renderer:
                env.render()
            if done:
                break


        env.reset()
        obs, __, __, __ = env.step([0,0,0,0,0,0,0])
        obs_vector = (round(obs['robot0_eef_pos'][0], 1), round(obs['robot0_eef_pos'][1], 1), round(obs['robot0_eef_pos'][2], 1)) # override
    
    
    if imshow:
        cap.release()
        cv2.destroyAllWindows()   
    return r

def observe(env, r, obs, imshow=True):
    img = Image.fromarray(obs["sideview_image"], 'RGB')
    result = r.vision_model(img)
    depth_img = obs["sideview_depth"]
    if imshow:
        renderings = result.render()
        cv2.imshow("Detections", renderings[0])
        key = cv2.waitKey(30) 
    centers = box_centers(result) # centers of a box for each detected object
    positions = positions_from_labelled_pixels(depth_img, env, centers)
    return positions
    

# Rewards based on how close the eef is to the cube, by grades
# So if there are 10 grades, moving from 1m away to .8m away gives 1 reward and 2 reward at approprate times.
def __reward(initial_distance, grades, eef_pos, cube_pos):
    distance = np.linalg.norm(eef_pos - cube_pos)

    for grade in grades.keys():
        if initial_distance / len(grades.keys()) * grade > distance and not grades[grade]:
            grades[grade] = True
            return len(grades.keys()) - grade
    return 0

def visualize(r):
    env = suite.make(
        env_name="Lift",
        robots="Kinova3",
        has_renderer=True, # the only difference
        has_offscreen_renderer=True,
        use_camera_obs=True,
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
    r = main(imshow=True)
    visualize(r)
