import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
from glob import glob
import argparse
import imageio.v3 as iio


def create_frame_npzstore(frame_idx, output_path, folder_1080, folder_540):
    """创建单帧的 NPZ 存储"""
    npz_path = f"{output_path}/frame{frame_idx:04d}.npz"
    data_dict = {}

    try:
        # 540p 分辨率数据
        data_dict['540'] = {}
        # 带采样维度的数据 [B,C,H,W,S]
        sample_shape = (1, 3, 540, 960, 8)  # 8个采样
        base_shape = (1, 3, 540, 960)

        datasets = ['color', 'diffuse', 'specular']
        for i, dataset_name in enumerate(datasets):
            data = np.zeros(sample_shape, dtype=np.float32)
            for j in range(8):
                exr_path = os.path.join(folder_540, f'{dataset_name}{i}.exr')
                # imageio 直接读取为RGB顺序
                exr_data = iio.imread(exr_path)[..., :3].astype(np.float32)
                exr_data = np.transpose(exr_data, (2, 0, 1))
                data[0, :, :, :, j] = exr_data
            data_dict['540'][dataset_name] = data

        # 其他的4维度数据
        datasets = ['albedo', 'depth', 'motion', 'normal', 'roughness']
        for i, dataset_name in enumerate(datasets):
            if dataset_name == 'depth':
                shape = (1, 1, 540, 960)
            elif dataset_name == 'motion':
                shape = (1, 2, 540, 960)
            else:
                shape = base_shape

            data = np.zeros(shape, dtype=np.float32)
            exr_path = os.path.join(folder_540, f'{dataset_name}.exr')
            exr_data = iio.imread(exr_path).astype(np.float32)
            
            if dataset_name == 'depth':
                data[0, 0] = exr_data[..., 0]
            elif dataset_name == 'motion':
                data[0, :2] = np.transpose(exr_data[..., :2], (2, 0, 1))
            else:
                data[0] = np.transpose(exr_data[..., :3], (2, 0, 1))

            data_dict['540'][dataset_name] = data

        # 1080p 分辨率数据
        data_dict['1080'] = {}
        hd_shape = (1, 3, 1080, 1920)
        datasets = ['albedo', 'depth', 'motion', 'normal', 'reference', 'roughness']
        for i, dataset_name in enumerate(datasets):
            if dataset_name == 'depth':
                shape = (1, 1, 1080, 1920)
            elif dataset_name == 'motion':
                shape = (1, 2, 1080, 1920)
            else:
                shape = hd_shape

            data = np.zeros(shape, dtype=np.float32)
            exr_path = os.path.join(folder_1080, f'{dataset_name}.exr')
            exr_data = iio.imread(exr_path).astype(np.float32)
            
            if dataset_name == 'depth':
                data[0, 0] = exr_data[..., 0]
            elif dataset_name == 'motion':
                data[0, :2] = np.transpose(exr_data[..., :2], (2, 0, 1))
            else:
                data[0] = np.transpose(exr_data[..., :3], (2, 0, 1))

            data_dict['1080'][dataset_name] = data
    finally:
        np.savez_compressed(npz_path, **data_dict)


def main(args):
    # 遍历所有 framexxxx 文件夹
    frame_folders = glob(os.path.join(args.base_folder, "frame*"))
    frame_folders = sorted(frame_folders)[args.start_frame:]
    os.makedirs(args.output_folder, exist_ok=True)
    for i, frame_folder in enumerate(frame_folders, start=args.start_frame):
        folder_1080 = os.path.join(frame_folder, "1080")
        folder_540 = os.path.join(frame_folder, "540")
        create_frame_npzstore(i, args.output_folder, folder_1080, folder_540)
        print(f"Processed frame {i}")


base_folder = "/data/hjy/realtimeds_raw/Sponza"
output_folder = "/data/yy/realtimeDS_npz/Sponza"
start_frame = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zarr array data')
    parser.add_argument('--base_folder', type=str, default=base_folder,
                        help='Input data folder')
    parser.add_argument('--output_folder', type=str, default=output_folder,
                        help='Npz output folder')
    parser.add_argument('--start_frame', type=int, default=start_frame,
                        help='Start frame number')

    args = parser.parse_args()

    main(args)

    print("Transform finished")
