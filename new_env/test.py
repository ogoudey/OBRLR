from envs.custom import Custom
import pygame

def take_teleop_input():
    action = [0.0,0.0,0.0,0.0,0,0,0]
    trigger = input("Button: ")
    if trigger == "q":
        action[0] = 0.1
    elif trigger == "w":
        action[1] = 0.1
    elif trigger == "e":
        action[2] = 0.1
    elif trigger == "r":
        action[6] = 0.1
    elif trigger == "a":
        action[0] = -0.1
    elif trigger == "s":
        action[1] = -0.1
    elif trigger == "d":
        action[2] = -0.1
    elif trigger == "f":
        action[6] = -0.1

    return action

env = Custom(
    robots="Kinova3",
    has_renderer=True,
    has_offscreen_renderer=True
)
obs = env.reset()



while True:
    action = take_teleop_input()
    obs, reward, done, info = env.step(action)
    if done:
        break
env.close()
