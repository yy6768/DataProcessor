import os
import zarr
import numpy as np
import imageio
from glob import glob
import argparse


def create_frame_zipstore(frame_idx, output_path, folder_1080, folder_540):
    """创建单帧的 ZipStore"""
    zip_path = f"{output_path}/frame{frame_idx:04d}.zip"
    store = zarr.ZipStore(zip_path, mode='w')
    root = zarr.group(store=store)
    
    try:
        # 540p 分辨率数据
        root.create_group('540')
        # 带采样维度的数据 [B,C,H,W,S]
        sample_shape = (1, 3, 540, 960, 8)  # 8个采样
        base_shape = (1, 3, 540, 960)

        datasets = ['color','diffuse','specular']
        for i, dataset_name in enumerate(datasets):
            data = np.zeros(sample_shape, dtype=np.float32)
            for j in range(8):
                exr_path = os.path.join(folder_540, f'{dataset_name}{i}.exr')
                exr_data = np.transpose(np.array(imageio.imread(exr_path, format='EXR'))[..., :3].astype(np.float32), (2, 0, 1))
                data[0, :, :, :, j] = exr_data
            root.create_dataset(f'540/{dataset_name}', 
                                shape=sample_shape,
                                chunks=(1, 3, 128, 128,8),
                                dtype='f4',
                                data=data)        
        # 其他的4维度数据
        datasets = ['albedo', 'depth', 'motion', 'normal', 'reference', 'roughness']
        for i, dataset_name in enumerate(datasets):
            data = np.zeros(base_shape, dtype=np.float32)
            exr_path = os.path.join(folder_540, f'{dataset_name}.exr')
            exr_data = np.transpose(np.array(imageio.imread(exr_path, format='EXR'))[..., :3].astype(np.float32), (2, 0, 1))
            data[0, :, :, :] = exr_data
            root.create_dataset(f'540/{dataset_name}', 
                                shape=base_shape,
                                chunks=(1, 3, 128, 128),
                                dtype='f4',
                                data=data)
        
        # 1080p 分辨率数据
        root.create_group('1080')
        hd_shape = (1, 3, 1080, 1920)
        datasets = ['albedo', 'depth', 'motion', 'normal', 'reference', 'roughness']
        for i, dataset_name in enumerate(datasets):
            data = np.zeros(hd_shape, dtype=np.float32)
            exr_path = os.path.join(folder_1080, f'{dataset_name}.exr')
            exr_data = np.transpose(np.array(imageio.imread(exr_path, format='EXR'))[..., :3].astype(np.float32), (2, 0, 1))
            data[0, :, :, :] = exr_data
            root.create_dataset(f'1080/{dataset_name}', 
                                shape=hd_shape,
                                chunks=(1, 3, 256, 256),
                                dtype='f4',
                                data=data)        
    finally:
        store.close()
# 定义存储路径
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将图像数据转换为Zarr格式')
    parser.add_argument('--base_folder', type=str, default=r"D:\Falcor\res\Sponza",
                        help='输入数据的基础文件夹路径')
    parser.add_argument('--output_folder', type=str, default="zipped_data",
                        help='Zarr输出文件夹路径')
    parser.add_argument('--start_frame', type=int, default=340,
                        help='开始处理的帧序号')
    
    args = parser.parse_args()
    
    # 遍历所有 framexxxx 文件夹
    frame_folders = glob(os.path.join(args.base_folder, "frame*"))
    i = 0
    for frame_folder in frame_folders:
        folder_1080 = os.path.join(frame_folder, "1080")
        folder_540 = os.path.join(frame_folder, "540")
        if i > args.start_frame:
            create_frame_zipstore(i, args.output_folder, folder_1080, folder_540)
        i += 1

    print("Transform finished")
