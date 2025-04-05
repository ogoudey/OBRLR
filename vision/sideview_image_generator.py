import numpy as np
import robosuite as suite
from robosuite.utils.placement_samplers import UniformRandomSampler

from PIL import Image


# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Kinova3",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    horizon = 1000,
    camera_heights=400,
    camera_widths=400,
    camera_names="sideview",
)

env.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=env.cube,
                x_range=[-0.3, 0.3],
                y_range=[-0.3, 0.3],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=env.table_offset,
                z_offset=0.01,
            )


num_photos = 10
for i in range(0, num_photos):
    env.reset()
    obs, reward, done, info = env.step([0,0,0,0,0,0,0])  # take action in the environment
    img = Image.fromarray(obs["sideview_image"], 'RGB')
    img_name = 'sideview'+str(i)+'.png'
    img.save('vision/data/Robosuite1/' + img_name)

