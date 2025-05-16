"""
ChatGPT's amazing find:
pip install pyrealsense2-beta==2.56.0.8430

This file will be used for 1. data collection and 2. real state representation and rewards.

1. taking RGB pictures for YOLO training set.
2. Feeding D into the detection after bounding box centers are calculated (in the manner of /vision/sim_vision).


This file maybe should be move to /vision.

The camera was working SO SMOOTHLY the first time it went. Now it's incredibly skippy. What happened?

"""
import cv2
import numpy as np
import pyrealsense2 as rs


def test():
    print("Testing...")
    pipeline = rs.pipeline()
    config = rs.config()
    # Enable depth + color streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    for _ in range(30):
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            break
        except RuntimeError:
            continue
    try:
        while True:
            # Wait for frames
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                print("Frame timeout, skipping this frame.")
                continue
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Convert depth to colormap for display
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # Show image
            cv2.imshow('RealSense RGB + Depth', images)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def sample_real_data(starting_i=0, num_photos=5):
    import time
    # change args for name
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    j = 1
    while j < num_photos:
        print("Round", j)
        photo_id = starting_i + j
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            n = 3
            print("Waiting", n, "seconds...")
            time.sleep(n)
        except RuntimeError:
            print("\tFrame timeout, skipping this frame.")
            continue
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())

        # save image as somename{i+j}
        
        cv2.imshow('RealSense RGB + Depth', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  
        j += 1
        
if __name__ == "__main__":
    #c = RealSense()
    #c.test()
    sample_real_data(0, 5)
    
    """
    Once enough is gathered, annotate the data, and retrain YOLO on it.
    """
    
