"""
ChatGPT's absolute snag:
pip install pyrealsense2-beta==2.56.0.8430
"""
import cv2
import numpy as np
import pyrealsense2 as rs


pipeline = rs.pipeline()
config = rs.config()

# Only enable depth + color streams
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
