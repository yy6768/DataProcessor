import numpy as np
import os
import glob

def process_npz_files(directory_path, frame_cnt):
    """
    处理指定目录下的npz文件，将RTXDI数据的最后一维替换为均值，并保存更新后的文件
    
    参数:
        directory_path: npz文件所在的目录路径
        frame_cnt: 要处理的帧数
    
    返回:
        处理后的数据列表
    """
    processed_data = []
    
    for i in range(frame_cnt):
        # 构建文件名 frame{0000-frame_cnt-1}.npz
        filename = os.path.join(directory_path, f"frame{i:04d}.npz")
        
        # 检查文件是否存在
        if not os.path.exists(filename):
            print(f"警告: 文件 {filename} 不存在")
            continue
        
        # 加载npz文件
        ds = np.load(filename,
                         allow_pickle=True)
        ds_lr = ds['540'].item()
        print(ds_lr.keys())
        
        
        # 检查是否包含所需的键
        if  'RTXDI' not in ds['540'].item():
            print(f"警告: 文件 {filename} 中没有找到 ds['540']['RTXDI']")
            continue
        
        # 获取RTXDI数据
        rtxdi_data = ds['540'].item()['RTXDI']
        
        # 检查形状是否符合预期 (1,3,540,960,8)
        if rtxdi_data.shape[-1] != 8:
            print(f"警告: 文件 {filename} 中的RTXDI数据形状不符合预期，实际形状: {rtxdi_data.shape}")
            continue
        
        # 计算最后一维的均值并替换
        mean_values = np.mean(rtxdi_data, axis=-1, keepdims=True)
        rtxdi_data_mean = np.repeat(mean_values, rtxdi_data.shape[-1], axis=-1)
        
        # 更新数据
        ds['540'].item()['RTXDI'] = rtxdi_data_mean
        
        # 保存更新后的文件
        np.savez(filename, **ds)
        
        processed_data.append(ds)
        print(f"已处理并保存文件 {filename}")
    
    return processed_data


if __name__ == "__main__":
    process_npz_files("G:/realtimeds_dataset/House_0407_1_test", 150)
