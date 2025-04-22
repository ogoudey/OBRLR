from vision import camera_utils as cu

import cv2
import numpy as np
import torch
from PIL import Image
from collections import defaultdict


image_id = 0

vision_model_path = 'vision/cube_eef_detector.pt'

class SimVision:
    def __init__(self):
        self.vision_model = torch.hub.load('ultralytics/yolov5', 'custom', path=vision_model_path, force_reload=True) # path is to the model originally in runs/expX/weights/
        self.cube_position, self.eef_position = None, None
        self.reset()
    
    def reset(self):
        cv2.destroyAllWindows()
        self.cube_position = np.array([0.2913, 0.2629, 1.3597])
        self.eef_position = np.array([0.2927, 0.30001, 1.2391])
        #self.cap = cv2.VideoCapture(0)
            
    def detect(self, env_image, env_depth, sim, no_cap=True):
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
            print("Showing detection?:", not no_cap)
        
        
        _3d_positions = self.positions_from_labelled_pixels(centers, env_depth, sim)
        euler_distance = self.cube_position - self.eef_position
        np_array = np.concatenate((_3d_positions, euler_distance))
        print("Detection:", np_array)
        return torch.tensor(np_array, dtype=torch.float32)

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
        """
        if "eef" in labelled_pixel_dict.keys():
            points_eef = cu.transform_from_pixels_to_world(np.array([labelled_pixel_dict["eef"]]), expanded_depth, camera_to_world_transform)
            if np.isnan(points_eef[0]).all() or points_eef[0][0] > 5: # arbitrary attempts at removing bad input... [nan] or 2e^35 (??)
                print("Bad input in array diverted.")
                eef_pos = self.eef_position
            else:
                eef_pos = points_eef[0]
            positions_array.extend(eef_pos)
            self.eef_position = eef_pos
            print("Setting eef:", self.eef_position)
        else:
            positions_array.extend(self.eef_position)
            print("No eef:", self.eef_position)
        if "cube" in labelled_pixel_dict.keys():    
            points_cube = cu.transform_from_pixels_to_world(np.array([labelled_pixel_dict["cube"]]), expanded_depth, camera_to_world_transform)
            if np.isnan(points_cube[0]).all(): # for some reason sometimes theres all Nan in the tensor
                print("[nan] array diverted.")
                cube_pos = self.cube_position
            else:   
                cube_pos = points_cube[0]                
            positions_array.extend(cube_pos)
            self.cube_position = cube_pos
            print("Setting cube:", self.cube_position)
        else:
            print("No cube:", self.cube_position)
            positions_array.extend(self.cube_position)
        return np.array(positions_array)
