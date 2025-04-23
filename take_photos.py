### A combination of movement and vision ###


import sys
import os
import time
import random
import threading
from PIL import Image

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient

from kortex_api.autogen.messages import Session_pb2, Base_pb2, Common_pb2
from kortex_api.autogen.messages.Common_pb2 import Empty
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient

from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, DeviceManager_pb2, VisionConfig_pb2

import cv2

# replace ip/user/pass as needed
rtsp_url = "rtsp://admin:admin@192.168.1.10/color"
cap = cv2.VideoCapture(rtsp_url)

#from kortex_api.autogen.client_stubs.CameraClientRpc import SessionClient
#from kortex_api.autogen.messages.Camera_pb2 import GetImageRequest



# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

# Actuator speed (deg/s)
SPEED = 5.0

# replace ip/user/pass as needed
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
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check

def example_move_to_start_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    constrained_joint_angles = Base_pb2.ConstrainedJointAngles()
    """
    resp = base.GetMeasuredJointAngles()
    print(resp)
    for joint in resp.joint_angles:
        print("Joint value:", joint.value)
    actuator_count = base.GetActuatorCount().count
    angles = [0.0] * actuator_count
    print(angles)
    angles = []
    for joint in resp.joint_angles:
        angles.append(joint.value)
    print(angles)
    """
    angles = [289.22265625, 0.8567417860031128, 79.26780700683594, 89.48311614990234, 177.61058044433594, 273.60845947265625, 98.24454498291016]
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

def example_send_joint_speeds(base, mvmt):

    joint_speeds = Base_pb2.JointSpeeds()

    actuator_count = base.GetActuatorCount().count
    # The 7DOF robot will spin in the same direction for 10 seconds
    if actuator_count == 7:

        i = 0
        for speed in mvmt:
            joint_speed = joint_speeds.joint_speeds.add()
            joint_speed.joint_identifier = i 
            joint_speed.value = speed
            joint_speed.duration = 0
            i = i + 1
        print ("Sending the joint speeds for 1 seconds...")
        base.SendJointSpeedsCommand(joint_speeds)
    
    


    return True

def take_and_save():
    
    ret, frame = cap.read()
    img = Image.fromarray(frame, 'RGB')
    img_name = 'onboard_camera'+str(random.random()*10000)+'.png'
    print("Saved image", img_name)
    img.save('real/data/onboard/' +img_name)
    if not ret:
        return
        
def main():
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        # vision

        # end vision
        
        
        base = BaseClient(router)

        example_move_to_start_position(base)
        moves_from_start = 0
        while True:
            try:
                mvmt = [random.randint(-SPEED, SPEED) for i in range(0, 7)]
                if moves_from_start > 9:
                    take_and_save()
                take_and_save()
                example_send_joint_speeds(base, mvmt)
                ret, frame = cap.read()
                time.sleep(0.1)
                
                
            except KeyboardInterrupt:

                base.Stop()
                if input("Initial Position?") == "n":
                    return 1
                example_move_to_start_position(base)
                moves_from_start = 0
        print("Exiting")        
        cap.release()
        return 1

if __name__ == "__main__":
    exit(main())
