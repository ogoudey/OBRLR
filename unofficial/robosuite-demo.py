import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Kinova3",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_heights=40,
    camera_widths=40,
    camera_names="frontview"
)

# reset the environment
env.reset()

def policy(obs_vector):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    return action

obs, reward, done, info = env.step([0,0,0,0,0,0,0])  # take action in the environment

print(type(obs["frontview_image"]))
for i in range(1000):
    action = policy(obs)

    #action = [float(input('action[0]')), 0, 0, 0, 0, 0, 0]
    #print(action)
    obs, reward, done, info = env.step(action)  # take action in the environment
    if done:
        break
    env.render()  # render on display
