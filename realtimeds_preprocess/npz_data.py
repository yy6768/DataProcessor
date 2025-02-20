import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
from glob import glob
import argparse
import imageio.v3 as iio


def create_cropped_frame_npzstore(frame_idx, output_path, folder_1080, folder_540, offset_file):
    """创建裁剪后单帧的 NPZ 存储"""
    npz_path = f"{output_path}/frame{frame_idx:04d}.npz"
    data_dict = {}

    try:
        # 读取offset信息
        with open(offset_file, 'r') as f:
            offset_x, offset_y = map(int, f.read().strip().split(','))
        
        # 读取原始frameId
        frame_id_file = os.path.join(os.path.dirname(offset_file), 'frameId.txt')
        with open(frame_id_file, 'r') as f:
            original_frame_id = int(f.read().strip())
            
        # 添加位置信息到数据字典
        data_dict['meta'] = {
            'offset_x': offset_x,
            'offset_y': offset_y,
            'original_frame_id': original_frame_id
        }

        # 540p 分辨率数据
        data_dict['540'] = {}
        # 带采样维度的数据 [B,C,H,W,S]
        sample_shape = (1, 3, 128, 128, 8)  # 8个采样，裁剪后的尺寸
        base_shape = (1, 3, 128, 128)

        # 处理多采样数据
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

        # 处理其他4维度数据
        datasets = ['albedo', 'depth', 'motion', 'normal']
        for i, dataset_name in enumerate(datasets):
            if dataset_name == 'depth':
                shape = (1, 1, 128, 128)
            elif dataset_name == 'motion':
                shape = (1, 2, 128, 128)
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
        hd_shape = (1, 3, 256, 256)  # 裁剪后的高分辨率尺寸
        # datasets = ['albedo', 'depth', 'motion', 'normal', 'reference']
        datasets = ['reference']
        for i, dataset_name in enumerate(datasets):
            if dataset_name == 'depth':
                shape = (1, 1, 256, 256)
            elif dataset_name == 'motion':
                shape = (1, 2, 256, 256)
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
    frame_folders = sorted(frame_folders)
    os.makedirs(args.output_folder, exist_ok=True)
    
    for frame_folder in frame_folders:
        # 从文件夹名称中提取帧索引
        frame_idx = int(os.path.basename(frame_folder).replace('frame', ''))
        
        folder_1080 = os.path.join(frame_folder, "1080")
        folder_540 = os.path.join(frame_folder, "540")
        offset_file = os.path.join(frame_folder, "offset.txt")
        
        create_cropped_frame_npzstore(
            frame_idx, 
            args.output_folder, 
            folder_1080, 
            folder_540, 
            offset_file
        )
        print(f"Processed frame {frame_idx}")


base_folder = "/data/hjy/realtimeds_cropped/Sponza"
output_folder = "/data/yy/realtimeDS_npz/Sponza_npz"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cropped data to NPZ')
    parser.add_argument('--base_folder', type=str, default=base_folder,
                      help='Input cropped data folder')
    parser.add_argument('--output_folder', type=str, default=output_folder,
                      help='Npz output folder')

    args = parser.parse_args()
    main(args)
    print("Transform finished")