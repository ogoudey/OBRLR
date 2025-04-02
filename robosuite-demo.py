import numpy as np
import robosuite as suite

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

def policy(obs_vector):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    return action

for i in range(1000):
    action = policy(obs_vector)

    #action = [float(input('action[0]')), 0, 0, 0, 0, 0, 0]
    #print(action)
    obs, reward, done, info = env.step(action)  # take action in the environment
    print(obs, reward, done, info)
    env.render()  # render on display
