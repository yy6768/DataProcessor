"""
From Nppd
Compute depth from position and camera_position
"""
import numpy as np
from tqdm import tqdm
import os
import OpenEXR
import Imath
import zarr

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
            camera_position_path = os.path.join(frame_path, "camera_position")
            camera_position_zarr = zarr.open(camera_position_path, mode = 'r')
            camera_position = np.array(camera_position_zarr)
            
            # position
            position_path = os.path.join(frame_path, "position")
            z = zarr.open(position_path, mode="r")
            data = np.array(z)
            position = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[3]), dtype=np.float32)
            position[0] = (data[0]).astype(np.float32)
            position[1] = (data[1]).astype(np.float32)
            position[2] = (data[2]).astype(np.float32)

            # Compute depth
            # print('\n')
            # print(f"position: {position.shape}")
            # print(f"camera_position: {camera_position.shape}")
            depth_data = depth(position, camera_position)
            # print(f"depth: {depth_data.shape}")
            
            # output exr file
            _, height, width, _ = depth_data.shape
    
            header = OpenEXR.Header(width, height)
            header["channels"] = {
                "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            }
            
            depth_path = os.path.join(f"/data/hjy/exrset_test/output/{scene}/{frame}", "depth.exr")
            # print(depth_path)
            exr_file = OpenEXR.OutputFile(depth_path, header)
            depth_data = np.mean(depth_data, axis=-1)
            depth_min = depth_data.min()
            depth_max = depth_data.max()
            depth_data = (depth_data - depth_min) / (depth_max - depth_min)
            
            R = depth_data[0, :, :].astype(np.float32).tobytes()
            G = depth_data[0, :, :].astype(np.float32).tobytes()
            B = depth_data[0, :, :].astype(np.float32).tobytes()
            
            exr_file.writePixels({"R": R, "G": G, "B": B})
            exr_file.close()
            
ComputeDepth("/data/hjy/exrset_test/unzip_file")