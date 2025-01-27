# -----------------------------------------------------------------------------------
# Company: TensorSense
# Project: LaserGaze
# File: main.py
# Description: This script demonstrates a basic example of how to use the GazeProcessor class
#              from the LaserGaze project. It sets up the gaze detection system with
#              optional visualization settings and an asynchronous callback for processing
#              gaze vectors. The example provided here can be modified or extended by
#              contributors to fit specific needs or to experiment with different settings
#              and functionalities. It serves as a starting point for developers looking to
#              integrate and build upon the gaze tracking capabilities provided by the
#              GazeProcessor in their own applications.
# Author: Sergey Kuldin
# -----------------------------------------------------------------------------------

from GazeProcessor import GazeProcessor
from VisualizationOptions import VisualizationOptions
import asyncio

# async def gaze_vectors_collected(left_vec, right_vec, left_center, right_center):
#     print(
#         f"left vector: {left_vec}, right vector: {right_vec}, left_center: {left_center}, right_center: {right_center}")
#     # print(f"left_center: {left_center[0]}")
async def gaze_vectors_collected(left_world_coords, right_world_coords):
    print(f"Left eye world coordinates (mm): {left_world_coords}")
    print(f"Right eye world coordinates (mm): {right_world_coords}")


async def main():
    vo = VisualizationOptions()
    gp = GazeProcessor(visualization_options=vo, callback=gaze_vectors_collected)
    await gp.start()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()