import numpy as np
import robosuite as suite

import random
from collections import defaultdict


# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Kinova3",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
)

# reset the environment
env.reset()

actions = [ all combinations of -1,0,1 in a seven length vector]
qtable = defaultdict(lambda : {{-0.1:0.0, 0.0:0.0, 0.1:0.0})

def policy(obs_vector, epsilon=0.1):
    
    if random.random() < epsilon:
        for joint in env.action_spec[0]:
            action = []
            action += random.choice([-0.1, 0.0, 0.1])
    #action = np.random.randn(*env.action_spec[0].shape) * 0.1
    print(action)
    return action

for i in range(1000):
    action = policy(obs_vector)
    
    #action = [float(input('action[0]')), 0, 0, 0, 0, 0, 0]
    #print(action)
    obs, reward, done, info = env.step(action)  # take action in the environment
    print(obs, reward, done, info)
    env.render()  # render on display
       

