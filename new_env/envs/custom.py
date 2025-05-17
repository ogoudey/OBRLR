import os

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import MujocoXMLObject
from robosuite.environments.manipulation.stack import TableArena  # same arena as Lift uses
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.tasks import ManipulationTask
from robosuite.environments.manipulation.stack import UniformRandomSampler 

class Custom(Lift):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def add_cup(self, all_objects):
        cup_path = "assets/objects/cup.xml"
        cup = MujocoXMLObject(name="cup", fname=cup_path)
        all_objects.append(cup)
        return cup
    
    def _load_model(self):

        super()._load_model()


        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # 3. Build the arena
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        all_objects = [self.cube]
        self.cup = self.add_cup(all_objects)
        
        
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=all_objects,
            x_range=[-0.2, 0.2],
            y_range=[-0.2, 0.2],
            rotation=None,
            ensure_object_boundary_in_range=False,  # keep this False or True as you wish
            ensure_valid_placement=False,            # ← allow overlapping spawns
            reference_pos=self.table_offset,
            z_offset=0.01,
        )
        # Hook into the placement initializer so it will spawn your cup
        #all_objects = [self.cube, self.cup]
        #if self.placement_initializer is not None:
        #    self.placement_initializer.reset()
        #    self.placement_initializer.add_objects(all_objects)
        

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=all_objects,
        )
