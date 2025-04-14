from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient

from kortex_api.autogen.messages import Session_pb2, Base_pb2, Common_pb2

import sys
import os
import threading
import time
import json

def move_to_start_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    constrained_joint_angles = Base_pb2.ConstrainedJointAngles()

    actuator_count = base.GetActuatorCount().count
    angles = [0.0] * actuator_count

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
    TIMEOUT_DURATION = 20
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Joint angles reached")
    else:
        print("Timeout on action notification wait")
    return finished

def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check

def move_vector(base, vector):
    joint_speeds = Base_pb2.JointSpeeds()
    period = 1
    i = 0
    for vel in vector:
        joint_speed = joint_speeds.joint_speeds.add()
        joint_speed.joint_identifier = i 
        joint_speed.value = vel
        joint_speed.duration = 0
        i = i + 1
    print ("Sending the joint speeds for", period, "seconds...")
    base.SendJointSpeedsCommand(joint_speeds)
    



def rl():
    for iteration in range(0, num_iterations):
        for action_step in range(0, action_steps):
            # replay memory adding
            pass
        for gradient_step in range(0, gradient_steps):
            
    # robosuite training
    # => poliicy, parameters
    # attempt irl
    pass
    
def main():
    # Based on example 102.../04-send_joint_speeds
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        print("Movement test")
        # Create required services
        base = BaseClient(router)
        move_to_start_position(base)
        
        # ... somehow...
        #move_vector(base, [20, 10, 0, -10, -20, 0, 0])
        
        #rl()

def io():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities
    args = utilities.parseConnectionArguments() 
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        while(True):
            action = json.loads(input("Action"))
            move_vector(base, action)
            time.sleep(1)
            base.Stop()
    
if __name__ == "__main__":
    #io()
    main()
