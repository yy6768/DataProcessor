import os
import zarr
import numpy as np
import imageio.v2 as imageio
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
                                chunks=(1, 3, 128, 128, 8),
                                dtype='f4',
                                data=data)        
        # 其他的4维度数据
        datasets = ['albedo', 'depth', 'motion', 'normal', 'roughness']
        for i, dataset_name in enumerate(datasets):
            if dataset_name == 'depth':
                shape = (1, 1, 540, 960)
                chunks = (1, 1, 128, 128)
            elif dataset_name == 'motion':
                shape = (1, 2, 540, 960)
                chunks = (1, 2, 128, 128)
            else:
                shape = base_shape
                chunks = (1, 3, 128, 128)
                
            data = np.zeros(shape, dtype=np.float32)
            exr_path = os.path.join(folder_540, f'{dataset_name}.exr')
            exr_data = np.array(imageio.imread(exr_path, format='EXR')).astype(np.float32)
            if dataset_name == 'depth':
                data[0, 0] = exr_data[..., 0]
            elif dataset_name == 'motion':
                data[0, :2] = np.transpose(exr_data[..., :2], (2, 0, 1))
            else:
                data[0] = np.transpose(exr_data[..., :3], (2, 0, 1))
            
            root.create_dataset(f'540/{dataset_name}', 
                              shape=shape,
                              chunks=chunks,
                              dtype='f4',
                              data=data)

        # 1080p 分辨率数据
        root.create_group('1080')
        hd_shape = (1, 3, 1080, 1920)
        datasets = ['albedo', 'depth', 'motion', 'normal', 'reference', 'roughness']
        for i, dataset_name in enumerate(datasets):
            if dataset_name == 'depth':
                shape = (1, 1, 1080, 1920)
                chunks = (1, 1, 256, 256)
            elif dataset_name == 'motion':
                shape = (1, 2, 1080, 1920)
                chunks = (1, 2, 256, 256)
            else:
                shape = hd_shape
                chunks = (1, 3, 256, 256)
                
            data = np.zeros(shape, dtype=np.float32)
            exr_path = os.path.join(folder_1080, f'{dataset_name}.exr')
            exr_data = np.array(imageio.imread(exr_path, format='EXR')).astype(np.float32)
            if dataset_name == 'depth':
                data[0, 0] = exr_data[..., 0]
            elif dataset_name == 'motion':
                data[0, :2] = np.transpose(exr_data[..., :2], (2, 0, 1))
            else:
                data[0] = np.transpose(exr_data[..., :3], (2, 0, 1))
            
            root.create_dataset(f'1080/{dataset_name}', 
                              shape=shape,
                              chunks=chunks,
                              dtype='f4',
                              data=data)        
    finally:
        store.close()

def main(args):
    # 遍历所有 framexxxx 文件夹
    frame_folders = glob(os.path.join(args.base_folder, "frame*"))
    frame_folders = sorted(frame_folders)[args.start_frame:]
    for i, frame_folder in enumerate(frame_folders, start=args.start_frame):
        folder_1080 = os.path.join(frame_folder, "1080")
        folder_540 = os.path.join(frame_folder, "540")
        create_frame_zipstore(i, args.output_folder, folder_1080, folder_540)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zarr array data')
    parser.add_argument('--base_folder', type=str, default="/data/hjy/realtimeds_raw/BistroExterior",
                        help='Input data folder')
    parser.add_argument('--output_folder', type=str, default="../test_realtimeDS_data/BistroExterior",
                        help='Zarr output folder')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Start frame number')
    
    args = parser.parse_args()
    
    main(args)

    print("Transform finished")
