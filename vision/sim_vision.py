from vision import camera_utils as cu

import numpy as np
import torch
from PIL import Image
from collections import defaultdict

class SimVision:
    def __init__(self):
        self.vision_model = torch.hub.load('ultralytics/yolov5', 'custom', path='vision/best.pt', force_reload=True) # path is to the model originally in runs/expX/weights/
        self.cube_position = np.array([0.0, 0.0, 0.0])
        self.eef_position = np.array([1.0, 1.0, 1.0])
        
    def detect(self, env_image, env_depth, sim, no_cap=True):

        img = Image.fromarray(env_image, 'RGB')
        result = self.vision_model(img)
        if not no_cap:
            renderings = result.render()
            cv2.imshow("Detections", renderings[0])
            key = cv2.waitKey(30)  

        
        centers = self.box_centers(result)
        _3d_positions = self.positions_from_labelled_pixels(centers, env_depth, sim)
        euler_distance = self.cube_position - self.eef_position
        np_array = np.concatenate((_3d_positions, euler_distance))
        return torch.tensor(np_array, dtype=torch.float32)

    def box_centers(self, result):
        centers = dict()
        box_per_obj_cat = defaultdict(lambda : False) # Right now we are gauranteed one box per item. Not necesarily the best box...
        for detection in result.xywh[0]:
            if not box_per_obj_cat[detection[5]]:
                centers[result.names[detection[5].item()]] = (detection[1].item(), detection[2].item())
        

        return centers   
        
    def positions_from_labelled_pixels(self, labelled_pixel_dict, depth, sim):
        object_3d_positions = dict()
        expanded_depth = depth[np.newaxis, ...]

        world_to_pixel_transform = cu.get_camera_transform_matrix(sim, "sideview", 400, 400)
        camera_to_world_transform = np.linalg.inv(world_to_pixel_transform)
        positions_array = []
        # Definite format for the labelled pixels (input to a nn)
        """
        for obj in labelled_pixel_dict.keys():
            points = cu.transform_from_pixels_to_world(np.array([labelled_pixel_dict[obj]]), expanded_depth, camera_to_world_transform)
            object_3d_positions[obj] = points[0]
        """
        if "eef" in labelled_pixel_dict.keys():
            points_eef = cu.transform_from_pixels_to_world(np.array([labelled_pixel_dict["eef"]]), expanded_depth, camera_to_world_transform)
            positions_array.extend(points_eef[0])
            self.eef_position = points_eef[0]
        else:
            positions_array.extend(self.eef_position)
        if "cube" in labelled_pixel_dict.keys():    
            points_cube = cu.transform_from_pixels_to_world(np.array([labelled_pixel_dict["cube"]]), expanded_depth, camera_to_world_transform)
            positions_array.extend(points_cube[0])
            self.cube_position = points_cube[0]
        else:
            positions_array.extend(self.cube_position)
        return torch.tensor(positions_array, dtype=torch.float32)     
