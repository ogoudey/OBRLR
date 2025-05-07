#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time
import threading
import numpy as np
import cv2
import math
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2, Session_pb2


from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Session_pb2 

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

# onboard camera stuff
rtsp_url = "rtsp://admin:admin@192.168.1.10/color"
cap = cv2.VideoCapture(rtsp_url)
# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT \
        or notification.action_event == 11:
            e.set()
    return check

 

def cartesian_action_movement(base, base_cyclic, delta):
    
    print("Starting Cartesian action movement ...", delta)
    action = Base_pb2.Action()
    action.name = "Cartesian action movement"
    action.application_data = ""

    feedback = base_cyclic.RefreshFeedback()

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = feedback.base.tool_pose_x + delta[0]
    cartesian_pose.y = feedback.base.tool_pose_y + delta[1]
    cartesian_pose.z = feedback.base.tool_pose_z + delta[2]
    cartesian_pose.theta_x = feedback.base.tool_pose_theta_x# (degrees)
    cartesian_pose.theta_y = feedback.base.tool_pose_theta_y # (degrees)
    cartesian_pose.theta_z = feedback.base.tool_pose_theta_z# (degrees)

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def move_to_start_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    constrained_joint_angles = Base_pb2.ConstrainedJointAngles()
    
    
    # From sim: [0.0, math.pi/4, 0.0, math.pi/2, 0.0, math.pi/4, -math.pi/2]
    
    angles = [0.0, 45.0, 0.0, 90.0, 0.0, 45.0, -90.0]
    # Actuator 4 at 90 degrees
    for joint_id in range(len(angles)):
        joint_angle = constrained_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = angles[joint_id]
    
    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Reaching joint angles...")
    base.PlayJointTrajectory(constrained_joint_angles)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Joint angles reached")
    else:
        print("Timeout on action notification wait")
    return finished

def take_and_save():
    
    ret, frame = cap.read()
    img = Image.fromarray(frame, 'RGB')
    img_name = 'onboard_camera'+str(random.random()*10000)+'.png'
    print("Saved image", img_name)
    img.save('real/data/onboard/' +img_name)
    if not ret:
        return

class Real:
    def __init__(self, router, composition=None):

        self.base = BaseClient(router)
        self.base_cyclic = BaseCyclicClient(router)
        
        if composition == "reset_eef":
            self.reset_eef()
            self.send_gripper_speed_command(-1.0)            
        if composition == "midway_eef":
            self.send_gripper_speed_command(1.0)
        
    def really_do(self, action):
        action = action.detach().numpy() / 100 # [-1,1] is simple not the right coordinate frame
        print(action)
        cartesian_action_movement(self.base, self.base_cyclic, action[0:3])    
           
    def reset_eef(self):
        move_to_start_position(self.base)
        
    def send_gripper_speed_command(self, speed, duration=0.5):
        # Should this be wrapped?
        gripper_cmd = Base_pb2.GripperCommand()
        gripper_cmd.mode = Base_pb2.GRIPPER_SPEED
        finger = gripper_cmd.gripper.finger.add()
        finger.finger_identifier = 0
        finger.value = -1 * speed # To be consistent with sim

        # Send repeatedly during the duration to maintain command
        start_time = time.time()
        while time.time() - start_time < duration:
            self.base.SendGripperCommand(gripper_cmd)
            time.sleep(0.01)  # ~100Hz loop

        # Stop gripper after duration
        finger.value = 0.0
        self.base.SendGripperCommand(gripper_cmd)
    
def main2():
    r = Real()
    r.kill() 

    
def main():
    
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        # Create required services
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        # Example core
        success = True

        move_to_start_position(base)
        
        speed = .1
        while True:
            action = np.array([0.0,0.0,0.0,0.0])
            trigger = input("Button: ")
            if trigger == "q":
                action[0] = speed
            elif trigger == "w":
                action[1] = speed
            elif trigger == "e":
                action[2] = speed
            elif trigger == "r":
                action[3] = speed
            elif trigger == "p":
                num_photos = int(input("# photos to take: "))
            elif trigger == "c":
                example_move_to_start_position(base)
            else:
                try:
                    speed = float(trigger)
                    print("Assigning speed!")
                except ValueError:
                    print("OOPS!")
                    continue   
            cartesian_action_movement(base, base_cyclic, action[0:3])




        # You can also refer to the 110-Waypoints examples if you want to execute
        # a trajectory defined by a series of waypoints in joint space or in Cartesian space

        return 0 if success else 1

if __name__ == "__main__":
    exit(main2())
