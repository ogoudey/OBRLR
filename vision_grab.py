import numpy as np
import robosuite as suite
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

def detect():
    cap = cv2.VideoCapture(0)      
    env = suite.make(
        env_name="Lift",
        robots="Kinova3",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        horizon = 1000,
        camera_heights=400,
        camera_widths=400,
        camera_names="sideview",
    )

    r = Robot()
    while True:
        obs, reward, done, info = env.step(r.policy(None))  # take action in the environment   
        img = Image.fromarray(obs["sideview_image"], 'RGB')
        results = r.vision_model(img).render()
        cv2.imshow("Detections", results[0])
        key = cv2.waitKey(30)  
        if key & 0xFF == 27:  # ESC key to break out of the loop
            break

def main(imshow=True):
    env = suite.make(
        env_name="Lift",
        robots="Kinova3",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        horizon = 1000,
        camera_heights=400,
        camera_widths=400,
        camera_names="sideview",
    )

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
        obs_vector = (round(obs['robot0_eef_pos'][0], 1), round(obs['robot0_eef_pos'][1], 1), round(obs['robot0_eef_pos'][2], 1)) # Change to some non-linear function of pixels
        
        initial_distance = np.linalg.norm(obs['robot0_eef_pos'] - obs['cube_pos'])
        
        for i in range(0, episode_length):
            action = r.policy(obs_vector)
            obs, reward, done, info = env.step(action)  # take action in the environment
            if imshow:
                img = Image.fromarray(obs["sideview_image"], 'RGB')
                results = r.vision_model(img).render()
                cv2.imshow("Detections", results[0])
                key = cv2.waitKey(30) 
                if key & 0xFF == 27:
                    break
            
            reward = __reward(initial_distance, grades, obs['robot0_eef_pos'], obs['cube_pos'])

            if i == episode_length - 1:
                done = True

            obs_vector = (round(obs['robot0_eef_pos'][0], 1), round(obs['robot0_eef_pos'][1], 1), round(obs['robot0_eef_pos'][2], 1))
             

            if done:
                break


        env.reset()
        obs, __, __, __ = env.step([0,0,0,0,0,0,0])
        obs_vector = (round(obs['robot0_eef_pos'][0], 1), round(obs['robot0_eef_pos'][1], 1), round(obs['robot0_eef_pos'][2], 1)) # override
    
    
    if imshow:
        cap.release()
        cv2.destroyAllWindows()   
    return r


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
