"""
ChatGPT's absolute snag:
pip install pyrealsense2-beta==2.56.0.8430
"""

import pyrealsense2 as rs

pipe = rs.pipeline()
config = rs.config()

# Only enable depth + color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipe.start(config)

try:
    for i in range(100):
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        print("Frames arrived:", color_frame and depth_frame)
finally:
    pipe.stop()
