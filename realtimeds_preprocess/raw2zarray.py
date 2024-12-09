import os
import numpy as np
import zarr
from PIL import Image
import cv2
from tqdm import tqdm

def load_images_from_folder(folder, pattern='noisy'):
    """加载指定类型的图像"""
    images = []
    for filename in sorted(os.listdir(folder)):
        if pattern in filename:
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            images.append(img_array)
    return np.stack(images, axis=0) if images else None

def create_zarr_dataset(raw_path, zarr_path):
    # TODO: 添加配置类
    data_structure = {
        '540': {
            'noisy': {'shape': (8, 3, 540, 960, 8)},  # B,C,H,W,S
            'specular': {'shape': (8, 3, 540, 960, 8)},
            'diffuse': {'shape': (8, 3, 540, 960, 8)},
            'albedo': {'shape': (1, 3, 540, 960)},  # B,C,H,W
            'normal': {'shape': (1, 3, 540, 960)},
            'depth': {'shape': (1, 1, 540, 960)}
        },
        '1080': {
            'reference': {'shape': (1, 3, 1080, 1920)},
            'albedo': {'shape': (1, 3, 1080, 1920)},
            'motion_vector': {'shape': (1, 2, 1080, 1920)},
            'normal': {'shape': (1, 3, 1080, 1920)},
            'depth': {'shape': (1, 1, 1080, 1920)}
        }
    }

    # 遍历所有帧
    for frame in tqdm(sorted(os.listdir(raw_path))):
        if not frame.startswith('frame'):
            continue

        frame_path = os.path.join(raw_path, frame)
        # 创建frame对应的zarr组
        frame_store = zarr.open(os.path.join(zarr_path, f"{frame}.zarr"), mode='w')

        # 处理每个分辨率
        for resolution in ['540', '1080']:
            res_path = os.path.join(frame_path, resolution)
            if not os.path.isdir(res_path):
                continue

            # 创建分辨率对应的组
            res_group = frame_store.create_group(resolution)

            # 处理该分辨率下的所有数据类型
            for data_type, config in data_structure[resolution].items():
                try:
                    if data_type == 'noisy': # TODO: specular, diffuse 也是
                        # 特殊处理noisy数据
                        data = load_images_from_folder(res_path, 'noisy')
                        if data is not None:
                            B, H, W, C = data.shape
                            S = 8
                            data = data.transpose(0, 3, 1, 2).reshape(*config['shape'])
                            res_group.create_dataset(data_type, data=data, chunks=True)
                    else:
                        # 处理其他类型数据
                        data = load_images_from_folder(res_path, data_type)
                        if data is not None:
                            if len(config['shape']) == 4:  # B,C,H,W格式
                                data = data.transpose(0, 3, 1, 2)
                            res_group.create_dataset(data_type, data=data, chunks=True)
                except Exception as e:
                    print(f"Error processing {data_type} in {frame}/{resolution}: {e}")

if __name__ == "__main__":
    raw_path = '/path/to/raw/data'
    zarr_path = '/path/to/zarr/data'
    create_zarr_dataset(raw_path, zarr_path)