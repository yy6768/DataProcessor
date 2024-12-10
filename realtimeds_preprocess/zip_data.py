import os
import zarr
import numpy as np
import imageio
from glob import glob

# 定义存储路径
base_folder = "test_data"  # 替换为实际的文件夹路径
zarr_output_folder = "zipped_data"  # 替换为输出文件夹路径

# 遍历所有 framexxxx 文件夹
frame_folders = glob(os.path.join(base_folder, "frame*"))

for frame_folder in frame_folders:
    # 获取 framexxxx 中的子文件夹 1080 和 540
    folder_1080 = os.path.join(frame_folder, "1080")
    folder_540 = os.path.join(frame_folder, "540")

    # 确保两个子文件夹都存在
    if os.path.exists(folder_1080) and os.path.exists(folder_540):
        # 假设文件是图像数据或数组数据，读取它们（这里假设为图像或.npy 文件）
        files_1080 = glob(os.path.join(folder_1080, "*"))
        files_540 = glob(os.path.join(folder_540, "*"))
        
        # 创建 Zarr 存储路径
        frame_name = os.path.basename(frame_folder)
        zarr_frame_folder = os.path.join(zarr_output_folder, frame_name)
        
        # 如果不存在，创建一个新的 Zarr 文件
        if not os.path.exists(zarr_frame_folder):
            os.makedirs(zarr_frame_folder)

        # 将数据转换为 Zarr 格式
        # 这里假设数据是图像文件，可以根据需求调整
        zarr_1080 = zarr.open(os.path.join(zarr_frame_folder, "1080"), mode='w', shape=(len(files_1080), 1080, 1920, 3), dtype='float32')
        zarr_540 = zarr.open(os.path.join(zarr_frame_folder, "540"), mode='w', shape=(len(files_540), 540, 960, 3), dtype='float32')

        # 假设图像文件或.npy 文件可以加载成 numpy 数组
        for i, file in enumerate(files_1080):
            data = np.array(imageio.imread(file, format='EXR'))[..., :3].astype(np.float32)  # 只取前三个通道
            zarr_1080[i] = data
        
        for i, file in enumerate(files_540):
            data = np.array(imageio.imread(file, format='EXR'))[..., :3].astype(np.float32)  # 只取前三个通道
            zarr_540[i] = data

print("所有数据已成功转化为 Zarr 格式")
