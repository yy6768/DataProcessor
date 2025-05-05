import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
from glob import glob
import argparse
import imageio.v3 as iio
import OpenEXR
import Imath

import concurrent.futures

def read_exr_data(exr_path, dataset_name):
    """使用 OpenEXR 读取数据，返回形状为 (H, W, C)"""
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # 获取通道数据并重塑为 (height, width)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = ['R', 'G', 'B']
    data = [
        np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
        .reshape((height, width))  # 修改reshape顺序
        for c in channels
    ]
    
    # 堆叠通道维度，得到 (H, W, C)
    exr_data = np.stack(data, axis=-1)
    # print(exr_data.shape)
    # assert exr_data.shape == (height, width, 3), f"形状异常: {exr_data.shape}"
    return exr_data

def create_test_frame_npzstore(frame_folder, args):
    """创建测试单帧的 NPZ 存储"""
    
    output_path = args.output_folder
    render_height = args.render_height
    render_width = args.render_width
    upscale_height = render_height * 2
    upscale_width = render_width * 2
    frame_idx = int(os.path.basename(frame_folder).replace('frame', ''))
    folder_upscale = os.path.join(frame_folder, f"{upscale_height}")
    folder_render = os.path.join(frame_folder, f"{render_height}")
    npz_path = f"{output_path}/frame{frame_idx:04d}.npz"
    data_dict = {}

    try:
        offset_x, offset_y = 0, 0
        original_frame_id = frame_idx
        pre_offset_x, pre_offset_y = 0, 0
        # 添加位置信息到数据字典
        data_dict['meta'] = {
            'offset': np.array([offset_x, offset_y], dtype=np.int32),
            'frame_index': np.array([original_frame_id], dtype=np.int32),
            'pre_offset': np.array([pre_offset_x, pre_offset_y], dtype=np.int32)
        }
        # 低清p 分辨率数据
        data_dict[f'{render_height}'] = {}
        # 带采样维度的数据 [B,C,H,W,S]
        sample_shape = (1, 3, render_height, render_width, 8)  # 8个采样，裁剪后的尺寸
        base_shape = (1, 3, render_height, render_width)

        # 处理多采样数据的函数
        def process_sample_dataset(dataset_info):
            dataset_name, j = dataset_info
            exr_path = os.path.join(folder_render, f'{dataset_name}{j}.exr')
            # imageio 直接读取为RGB顺序
            exr_data = read_exr_data(exr_path, dataset_name).astype(np.float32) 
            exr_data = exr_data * 0.45
            return np.transpose(exr_data, (2, 0, 1))
        
        # 处理其他4维度数据的函数
        def process_base_dataset(dataset_info):
            dataset_name, shape = dataset_info
            exr_path = os.path.join(folder_render, f'{dataset_name}.exr')
            exr_data = read_exr_data(exr_path, dataset_name).astype(np.float32)
            
            data = np.zeros(shape, dtype=np.float32)
            if dataset_name == 'depth':
                data[0, 0] = exr_data[..., 0]
            elif dataset_name == 'motion':
                data[0, :2] = np.transpose(exr_data[..., :2], (2, 0, 1))
            else:
                data[0] = np.transpose(exr_data[..., :3], (2, 0, 1))
            
            return dataset_name, data
        
        # 处理高分辨率数据的函数
        def process_hd_dataset(dataset_info):
            dataset_name, shape = dataset_info
            exr_path = os.path.join(folder_upscale, f'{dataset_name}.exr')
            exr_data = read_exr_data(exr_path, dataset_name).astype(np.float32)
            data = np.zeros(shape, dtype=np.float32)
            if dataset_name == 'depth':
                data[0, 0] = exr_data[..., 0]
            elif dataset_name == 'motion':
                data[0, :2] = np.transpose(exr_data[..., :2], (2, 0, 1))
            else:
                data[0] = np.transpose(exr_data[..., :3], (2, 0, 1)) * 0.45
            
            return dataset_name, data

        # 使用线程池处理多采样数据
        datasets = ['color', 'diffuse', 'specular', 'RTXDI']
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for dataset_name in datasets:
                data = np.zeros(sample_shape, dtype=np.float32)
                # 为每个采样创建任务
                sample_tasks = [(dataset_name, j) for j in range(8)]
                sample_results = list(executor.map(process_sample_dataset, sample_tasks))
                
                # 填充结果
                for j, result in enumerate(sample_results):
                    data[0, :, :, :, j] = result
                
                data_dict[f'{render_height}'][dataset_name] = data

        # 使用线程池处理其他4维度数据
        datasets_info = []
        for dataset_name in ['albedo', 'depth', 'motion', 'normal']:
            if dataset_name == 'depth':
                shape = (1, 1, render_height, render_width)
            elif dataset_name == 'motion':
                shape = (1, 2, render_height, render_width)
            else:
                shape = base_shape
            datasets_info.append((dataset_name, shape))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            base_results = list(executor.map(process_base_dataset, datasets_info))
            for dataset_name, data in base_results:
                data_dict[f'{render_height}'][dataset_name] = data

        # 1080p 分辨率数据
        data_dict[f'{upscale_height}'] = {}
        hd_shape = (1, 3, upscale_height, upscale_width)  # 裁剪后的高分辨率尺寸
        
        # 使用线程池处理高分辨率数据
        datasets_info = []
        for dataset_name in ['reference']:
            # if dataset_name == 'depth':
            #     shape = (1, 1, upscale_height, upscale_width)
            # elif dataset_name == 'motion':
            #     shape = (1, 2, upscale_height, upscale_width)
            # else:
            #     shape = hd_shape
            shape = hd_shape
            datasets_info.append((dataset_name, shape))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            hd_results = list(executor.map(process_hd_dataset, datasets_info))
            for dataset_name, data in hd_results:
                data_dict[f'{upscale_height}'][dataset_name] = data
    finally:
        np.savez_compressed(npz_path, **data_dict)

def create_cropped_frame_npzstore(frame_folder, args):
    """创建裁剪后单帧的 NPZ 存储"""
    
    output_path = args.output_folder
    render_height = args.render_height
    render_width = args.render_width
    upscale_height = render_height * 2
    crop_height = args.crop_height
    crop_width = args.crop_width
    base_folder = os.path.dirname(frame_folder)
    frame_idx = int(os.path.basename(frame_folder).replace('frame', ''))
    offset_file = os.path.join(frame_folder, "offset.txt")
    folder_upscale = os.path.join(frame_folder, f"{upscale_height}")
    folder_render = os.path.join(frame_folder, f"{render_height}")
    id_file = os.path.join(frame_folder, "frameId.txt")
    npz_path = f"{output_path}/frame{frame_idx:04d}.npz"
    data_dict = {}

    try:
        # 读取offset信息
        with open(offset_file, 'r') as f:
            offset_x, offset_y = map(int, f.read().strip().split(','))
        
        # 读取原始frameId
        with open(id_file, 'r') as f:
            original_frame_id = int(f.read().strip())
        if original_frame_id != 0:
            pre_frame_folder = os.path.join(base_folder, f'frame{frame_idx - 1:04d}')
            pre_offset_file = os.path.join(pre_frame_folder, "offset.txt")
            with open(pre_offset_file, 'r') as f:
                pre_offset_x, pre_offset_y = map(int, f.read().strip().split(','))
        else:
            pre_offset_x, pre_offset_y = offset_x, offset_y
        # 添加位置信息到数据字典
        data_dict['meta'] = {
            'offset': np.array([offset_x, offset_y], dtype=np.int32),
            'frame_index': np.array([original_frame_id], dtype=np.int32),
            'pre_offset': np.array([pre_offset_x, pre_offset_y], dtype=np.int32)
        }
        # 低清 分辨率数据
        data_dict[f'{render_height}'] = {}
        # 带采样维度的数据 [B,C,H,W,S]
        sample_shape = (1, 3, crop_height, crop_width, 8)  # 8个采样，裁剪后的尺寸
        base_shape = (1, 3, crop_height, crop_width)

        # 多线程处理多采样数据
        def process_sample_dataset(dataset_info):
            dataset_name, j = dataset_info
            exr_path = os.path.join(folder_render, f'{dataset_name}{j}.exr')
            exr_data = read_exr_data(exr_path, dataset_name).astype(np.float32)
            return dataset_name, j, np.transpose(exr_data, (2, 0, 1))
        
        # 处理多采样数据
        datasets = ['color', 'RTXDI']
        sample_data = {}
        
        # 创建线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # 准备任务列表
            tasks = [(dataset_name, j) for dataset_name in datasets for j in range(8)]
            # 提交所有任务并获取结果
            results = list(executor.map(lambda x: process_sample_dataset(x), tasks))
            
            # 处理结果
            for dataset_name, j, data_result in results:
                if dataset_name not in sample_data:
                    sample_data[dataset_name] = np.zeros(sample_shape, dtype=np.float32)
                sample_data[dataset_name][0, :, :, :, j] = data_result
        
        # 将处理好的数据添加到数据字典
        for dataset_name, data in sample_data.items():
            data_dict[f'{render_height}'][dataset_name] = data

        # 多线程处理其他4维度数据
        def process_regular_dataset(dataset_name):
            if dataset_name == 'depth':
                shape = (1, 1, crop_height, crop_width)
            elif dataset_name == 'motion':
                shape = (1, 2, crop_height, crop_width)
            else:
                shape = base_shape

            data = np.zeros(shape, dtype=np.float32)
            exr_path = os.path.join(folder_render, f'{dataset_name}.exr')
            exr_data = read_exr_data(exr_path, dataset_name).astype(np.float32)
            
            if dataset_name == 'depth':
                data[0, 0] = exr_data[..., 0]
            elif dataset_name == 'motion':
                data[0, :2] = np.transpose(exr_data[..., :2], (2, 0, 1))
            else:
                data[0] = np.transpose(exr_data[..., :3], (2, 0, 1))
                
            return dataset_name, data
        
        # 处理其他4维度数据
        datasets = ['albedo', 'depth', 'motion', 'normal']
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_regular_dataset, datasets))
            for dataset_name, data in results:
                data_dict[f'{render_height}'][dataset_name] = data

        # 多线程处理1080p分辨率数据
        def process_hd_dataset(dataset_name):
            if dataset_name == 'depth':
                shape = (1, 1, 2 * crop_height, 2 * crop_width)
            elif dataset_name == 'motion':
                shape = (1, 2, 2 * crop_height, 2 * crop_width)
            else:
                shape = (1, 3, 2 * crop_height, 2 * crop_width)

            data = np.zeros(shape, dtype=np.float32)
            exr_path = os.path.join(folder_upscale, f'{dataset_name}.exr')
            exr_data = read_exr_data(exr_path, dataset_name).astype(np.float32)
            
            if dataset_name == 'depth':
                data[0, 0] = exr_data[..., 0]
            elif dataset_name == 'motion':
                data[0, :2] = np.transpose(exr_data[..., :2], (2, 0, 1))
            else:
                data[0] = np.transpose(exr_data[..., :3], (2, 0, 1))
                
            return dataset_name, data
        
        # 1080p 分辨率数据
        data_dict[f'{upscale_height}'] = {}
        datasets = ['reference']
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            results = list(executor.map(process_hd_dataset, datasets))
            for dataset_name, data in results:
                data_dict[f'{upscale_height}'][dataset_name] = data
    finally:
        np.savez_compressed(npz_path, **data_dict)


def main(args):
    # 遍历所有 framexxxx 文件夹
    frame_folders = glob(os.path.join(args.base_folder, "frame*"))
    frame_folders = sorted(frame_folders)
    os.makedirs(args.output_folder, exist_ok=True)
    frame_folders = frame_folders[60:60+120]
    for frame_folder in frame_folders:
        # 从文件夹名称中提取帧索引
        frame_idx = int(os.path.basename(frame_folder).replace('frame', ''))
        
        # create_cropped_frame_npzstore(
        #     frame_folder,
        #     args, 
        # )
        create_test_frame_npzstore(
            frame_folder,
            args, 
        )
        print(f"Processed frame {frame_idx}")


base_folder = "E:/RealtimeDS/data/breakfast_room"
output_folder = "G:/realtimeds_dataset/BreakfastRoom_0427_test"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cropped data to NPZ')
    parser.add_argument('--base_folder', type=str, default=base_folder,
                      help='Input cropped data folder')
    parser.add_argument('--output_folder', type=str, default=output_folder,
                      help='Npz output folder')
    parser.add_argument('--render_height', type=int, default=540,
                      help='Render height')
    parser.add_argument('--render_width', type=int, default=960,
                      help='Render width')
    parser.add_argument('--crop_height', type=int, default=128,
                      help='Crop height')
    parser.add_argument('--crop_width', type=int, default=128,
                      help='Crop width')

    args = parser.parse_args()
    main(args)
    print("Transform finished")
