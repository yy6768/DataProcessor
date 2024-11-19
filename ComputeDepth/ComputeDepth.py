import numpy as np
from tqdm import tqdm
import os
import OpenEXR
import Imath

# log depth
def depth(w_position, pos):
    """Computes per-sample compressed depth (disparity-ish)

    Args:
        w_position (ndarray, 3HWS): per-sample world-space positions
        pos (ndarray, size (3)): the camera's position in world-space
    
    Returns:
        motion (ndarray, 1HWS): per-sample compressed depth
    """
    # TODO: support any number of extra dimensions like apply_array
    d = np.linalg.norm(w_position - np.reshape(pos, (3, 1, 1, 1)), axis=0, keepdims=True)
    return d


def ComputeDepth(root_dir):
    for scene in tqdm(os.listdir(root_dir), desc = f"Processing scenes in {root_dir}"):
        scene_path = os.path.join(root_dir, scene)
        for frame in tqdm(os.listdir(scene_path), desc = f"Processing frames in {scene}"):
            frame_path = os.path.join(scene_path, frame)
            # camera_position
            camera_position_path = os.path.join(frame_path, "camera_position.txt")
            with open(camera_position_path, 'r') as file:
                data = file.readlines()
            camera_position = [float(line.strip()) for line in data]
            camera_position_array = np.array(camera_position)
            
            # position
            position_path = os.path.join(frame_path, "position.exr")
            exr_file = OpenEXR.InputFile(position_path)
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            channels = ['R', 'G', 'B']
            channel_data = [np.frombuffer(exr_file.channel(c), dtype=np.float32) for c in channels]
    
            channel_data = [c.reshape((height, width)) for c in channel_data]
            
            position_array = np.stack(channel_data, axis=0)
            
            # Compute depth
            depth_data = depth(position_array, camera_position_array)
            
            # output exr file
            _, _, height, width = depth_data.shape
    
            header = OpenEXR.Header(width, height)
            header["channels"] = {
                "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            }
            
            depth_path = os.path.join(frame_path, "depth.exr")
            exr_file = OpenEXR.OutputFile(depth_path, header)
            
            R = depth_data[:, 0, :, :].astype(np.float32).tobytes()
            G = depth_data[:, 1, :, :].astype(np.float32).tobytes()
            B = depth_data[:, 2, :, :].astype(np.float32).tobytes()
            
            exr_file.writePixels({"R": R, "G": G, "B": B})
            exr_file.close()
            
ComputeDepth("/data/hjy/exrset_test/output")