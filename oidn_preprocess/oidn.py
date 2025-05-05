import os
import subprocess
import numpy as np
import cv2
import re
from shutil import move
import OpenEXR
import Imath
import array

def read_exr(exr_path):
    """
    读取EXR文件并返回RGB图像
    """
    file = OpenEXR.InputFile(exr_path)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    # 读取三个通道
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R, G, B) = [array.array('f', file.channel(c, FLOAT)).tolist() for c in 'RGB']
    
    # 将数据重组为numpy数组
    img = np.zeros((size[1], size[0], 3), dtype=np.float32)
    img[:,:,0] = np.array(R).reshape(size[1], size[0])
    img[:,:,1] = np.array(G).reshape(size[1], size[0])
    img[:,:,2] = np.array(B).reshape(size[1], size[0])
    
    return img

def write_exr(img, exr_path):
    """
    将RGB图像写入EXR文件
    """
    height, width, _ = img.shape
    
    header = OpenEXR.Header(width, height)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = dict([(c, half_chan) for c in "RGB"])
    
    out = OpenEXR.OutputFile(exr_path, header)
    R = array.array('f', img[:,:,0].flatten().tolist()).tobytes()
    G = array.array('f', img[:,:,1].flatten().tolist()).tobytes()
    B = array.array('f', img[:,:,2].flatten().tolist()).tobytes()
    
    out.writePixels({'R': R, 'G': G, 'B': B})
    out.close()

def write_pfm(img, pfm_path):
    """
    将RGB图像写入PFM文件
    """
    height, width, channels = img.shape
    
    with open(pfm_path, 'wb') as f:
        # 写入PFM头部
        f.write(b'PF\n')
        f.write(f'{width} {height}\n'.encode())
        f.write(b'-1.0\n')  # 负数表示小端字节序
        
        # 写入数据（需要上下翻转）
        img_flipped = np.flip(img, 0).copy()
        img_flipped.tofile(f)

def read_pfm(pfm_path):
    """
    读取PFM文件并返回RGB图像
    """
    with open(pfm_path, 'rb') as f:
        # 读取头部
        header = f.readline().decode().strip()
        width, height = map(int, f.readline().decode().strip().split())
        scale = float(f.readline().decode().strip())
        
        # 确定数据类型和通道数
        channels = 3 if header == 'PF' else 1
        endian = '<' if scale < 0 else '>'
        
        # 读取数据
        data = np.fromfile(f, dtype=f'{endian}f')
        img = data.reshape((height, width, channels))
        
        # 如果scale为负，需要上下翻转图像
        if scale < 0:
            img = np.flip(img, 0)
            
        return img

def oidn_denoise(scene_path, oidn_path, resolution_height=1080):
    """
    对场景中的所有帧进行OIDN降噪处理
    
    参数:
    scene_path (str): 场景路径，包含帧文件夹
    oidn_path (str): OIDN可执行文件的路径
    resolution_height (int): 分辨率高度，默认为1080
    """
    # 获取所有帧文件夹
    frame_folders = [f for f in os.listdir(scene_path) if re.match(r'frame\d+', f)]
    frame_folders.sort()  # 确保按顺序处理
    
    if not frame_folders:
        raise ValueError(f"在路径 {scene_path} 中未找到帧文件夹")
    
    resolution_height_str = str(resolution_height)
    oidn_exe = os.path.join(oidn_path, "oidnDenoise.exe")
    
    if not os.path.exists(oidn_exe):
        raise FileNotFoundError(f"未找到OIDN可执行文件: {oidn_exe}")
    
    for frame_folder in frame_folders:
        frame_dir = os.path.join(scene_path, frame_folder, resolution_height_str)
        
        # 检查必要的文件是否存在
        reference_path = os.path.join(frame_dir, "reference.exr")
        albedo_path = os.path.join(frame_dir, "albedo.exr")
        normal_path = os.path.join(frame_dir, "normal.exr")
        
        if not all(os.path.exists(p) for p in [reference_path, albedo_path, normal_path]):
            print(f"警告: 在 {frame_dir} 中缺少必要的EXR文件，跳过此帧")
            continue
        
        # 重命名reference.exr为reference_old.exr
        reference_old_path = os.path.join(frame_dir, "reference_old.exr")
        move(reference_path, reference_old_path)
        
        # 创建PFM文件
        reference_pfm = os.path.join(frame_dir, "reference_old.pfm")
        albedo_pfm = os.path.join(frame_dir, "albedo.pfm")
        normal_pfm = os.path.join(frame_dir, "normal.pfm")
        output_pfm = os.path.join(frame_dir, "reference.pfm")
        
        # 读取EXR并写入PFM
        write_pfm(read_exr(reference_old_path), reference_pfm)
        write_pfm(read_exr(albedo_path), albedo_pfm)
        write_pfm(read_exr(normal_path), normal_pfm)
        
        # 调用OIDN进行降噪
        cmd = [
            oidn_exe,
            "-hdr", reference_pfm,
            "-alb", albedo_pfm,
            "-nrm", normal_pfm,
            "-o", output_pfm
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"成功对 {frame_dir} 进行降噪")
            
            # 将输出的PFM转换回EXR
            denoised_img = read_pfm(output_pfm)
            write_exr(denoised_img, reference_path)
            
            # 清理临时PFM文件
            for pfm_file in [reference_pfm, albedo_pfm, normal_pfm, output_pfm]:
                if os.path.exists(pfm_file):
                    os.remove(pfm_file)
                    
        except subprocess.CalledProcessError as e:
            print(f"在处理 {frame_dir} 时OIDN降噪失败: {e}")
            # 如果失败，恢复原始reference文件
            if os.path.exists(reference_old_path):
                move(reference_old_path, reference_path)

def process_scene(scene_path, oidn_path, resolution_height=1080):
    """
    处理单个场景的所有帧
    
    参数:
    scene_path (str): 场景路径
    oidn_path (str): OIDN可执行文件的路径
    resolution_height (int): 分辨率高度
    """
    print(f"开始处理场景: {scene_path}")
    oidn_denoise(scene_path, oidn_path, resolution_height)
    print(f"场景 {scene_path} 处理完成")

if __name__ == "__main__":
    # 设置OIDN可执行文件路径
    oidn_path = r"G:\huawei_realtimeds\oidn\bin"

    # 设置场景路径
    scene_path = r"E:\RealtimeDS\data\BMW_0417_2"
    # 设置分辨率高度
    resolution_height = 1080

    # 处理场景
    process_scene(scene_path, oidn_path, resolution_height)



