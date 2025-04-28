from vision import camera_utils as cu

import cv2
import numpy as np
import torch
from PIL import Image
from collections import defaultdict


image_id = 0

vision_model_path = 'vision/cube_eef_detector.pt'

class SimVision:
    def __init__(self, use_sim_camera=True):
        self.vision_model = torch.hub.load('ultralytics/yolov5', 'custom', path=vision_model_path, force_reload=True) # path is to the model originally in runs/expX/weights/
        self.cube_position, self.eef_position = None, None
        self.reset()
        self.use_sim_camera = use_sim_camera
    
    def reset(self):
        cv2.destroyAllWindows()
        self.cube_position = None
        self.eef_position = None
        #self.cap = cv2.VideoCapture(0)
    
            
    def detect(self, obs, sim, no_cap=True):
        
        if self.use_sim_camera:
            env_image = obs["sideview_image"]
            env_depth = obs["sideview_depth"]
            img = Image.fromarray(env_image, 'RGB')
            ### CHANGE FOR IMAGE DATA COLLECTION ###
            save_img_data = False
            if save_img_data:
                global image_id
                image_id += 1
                img_name = 'sideview'+str(image_id)+'.png'
                img.save('data/Robosuite2/' + img_name)
            ###         ###
            result = self.vision_model(img)
            
            centers = self.box_centers(result)
            #print("Centers:", centers)
            if not no_cap:
            
                rendering = result.render()[0].copy()

                for label, (x,y) in centers.items():
                    cv2.circle(rendering, (int(x), int(y)), radius=5, color=(255,255,255))         
                    cv2.putText(
                        rendering, label, (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA
                    )
                cv2.circle(rendering, (0, 0), radius=5, color=(0,122,255))
                cv2.circle(rendering, (399, 399), radius=5, color=(0,122,255))
                cv2.imshow("Detections", rendering)
                key = cv2.waitKey(5)  
            else:
                #print("Showing detection?:", not no_cap)
                pass
            
            
            _3d_positions = self.positions_from_labelled_pixels(centers, env_depth, sim)
        else: # no camera
            _3d_positions = obs['cube_pos']
            
        # show for teleop #
        if not no_cap and self.use_sim_camera:
            env_image = obs["sideview_image"]
            img = Image.fromarray(env_image, 'RGB')     
            cv2.imshow("Detections", img)
            key = cv2.waitKey(5) 
        distance = _3d_positions - sim.env.robots[0]._hand_pos['right']
        grasp = self.grasp_position(obs)
        pp = sim.env.robots[0]._hand_pos['right']
        #print(pp)
        #print(_3d_positions)
        #print(distance)
        #print(grasp)
        np_array = np.concatenate((pp, _3d_positions, distance, grasp))
        #print("Detection:", np_array, "(length:", len(np_array), ")")
        return torch.tensor(np_array, dtype=torch.float32)
    
    def grasp_position(self, obs):
        gripper_pos = obs["robot0_gripper_qpos"]    
        opening = gripper_pos.mean() 
        return np.array([opening])
    
    def box_centers(self, result):
        centers = dict()
        box_per_obj_cat = defaultdict(lambda : False) # Right now we are gauranteed one box per item. Not necesarily the best box...
        for detection in result.xywh[0]:
            #print(detection)
            if not box_per_obj_cat[detection[5]]:
                centers[result.names[detection[5].item()]] = (detection[0].item(), detection[1].item())
        

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
        
        if "eef" in labelled_pixel_dict.keys():
            points_eef = cu.transform_from_pixels_to_world(np.array([labelled_pixel_dict["eef"]]), expanded_depth, camera_to_world_transform)
            if np.isnan(points_eef[0]).any() or np.any((points_eef[0] < -2) | (points_eef[0] > 2)): # arbitrary attempts at removing bad input like [nan], -4e_20, or 2e^35 (??)
                print("Bad input in eef pos diverted: ", points_eef[0])
                eef_pos = self.eef_position
            else:
                eef_pos = points_eef[0]
            positions_array.extend(eef_pos)
            self.eef_position = eef_pos
            print("Setting eef:", self.eef_position)
        else:
            positions_array.extend(self.eef_position)
            print("No eef:", self.eef_position)
        """
        if "cube" in labelled_pixel_dict.keys():    
            points_cube = cu.transform_from_pixels_to_world(np.array([labelled_pixel_dict["cube"]]), expanded_depth, camera_to_world_transform)
            if np.isnan(points_cube[0]).any() or np.any((points_cube[0] < -2) | (points_cube[0] > 2)): # for some reason sometimes theres all Nan in the tensor
                #print("Bad input in cube pos diverted:", points_cube[0])
                cube_pos = self.cube_position
            else:   
                cube_pos = points_cube[0]                
            positions_array.extend(cube_pos)
            self.cube_position = cube_pos
            #print("Setting cube:", self.cube_position)
        else:
            #print("No cube:", self.cube_position)
            positions_array.extend(self.cube_position)
        return np.array(positions_array)
