import numpy as np
import torch
import torch.nn.functional as F
import OpenEXR
import Imath
import pyexr
import yaml
import skimage.metrics
from skimage.metrics import structural_similarity as ssim

with open('motion_vector_config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
BASE_DIR = config['BASE_DIR']
IS_USE_1080 = config['IS_USE_1080']
name = config['name']
BEGIN_FRAME = config['frame']['BEGIN_FRAME']
END_FRAME = config['frame']['END_FRAME']

def warp(img, motion_vector):
    """
    Warp image using motion vector
    """
    _, _, height, width = img.shape
    device = img.device
    motion_vector = torch.tensor(motion_vector, device=device)

    # 创建基础网格
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=img.device),
        torch.arange(width, device=img.device),
        indexing="ij",
    )
    # 应用运动向量
    grid_x = grid_x + motion_vector[:, :, 1] * width
    grid_y = grid_y + motion_vector[:, :, 0] * height

    # 归一化网格坐标到 [-1, 1] 范围
    grid_x = 2.0 * grid_x / (width - 1) - 1.0
    grid_y = 2.0 * grid_y / (height - 1) - 1.0

    # 堆叠网格坐标
    grid = torch.stack((grid_x, grid_y), dim=2)
    grid = grid.unsqueeze(0)
    result = F.grid_sample(
        img,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return result, grid

def check_motion_vector(result, next_ref): #计算了result和next_ref的SSIM和PSNR
    """
    检查motion vector是否正确
    """
    #将tensor转换为RGB_numpy
    result = result.squeeze(0)
    next_ref = next_ref.squeeze(0)
    rgb_result = result[:3, :, :]
    rgb_next_ref = next_ref[:3, :, :]
    rgb_result = rgb_result.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    rgb_next_ref = rgb_next_ref.permute(1, 2, 0).cpu().numpy().astype(np.float32)

    #归一化
    rgb_result = rgb_result/np.max(rgb_result)
    rgb_next_ref = rgb_next_ref/np.max(rgb_next_ref)
    #print(rgb_result.shape,rgb_next_ref.shape)

    # 计算 SSIM
    ssim_value, _ = ssim(rgb_result,rgb_next_ref,channel_axis=-1,data_range=1.0, full=True)

    # 计算 PSNR
    psnr = skimage.metrics.peak_signal_noise_ratio(rgb_result,rgb_next_ref)
    print(ssim_value,psnr)
    pass

def load_motion_vector(motion_vector_path):
    # 打开 EXR 文件
    exr = OpenEXR.InputFile(motion_vector_path)

    # 获取图像尺寸
    dw = exr.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # 读取 R 和 G 通道
    HALF = Imath.PixelType(Imath.PixelType.HALF)
    r_str = exr.channel('R', HALF)  # 获取 R 通道数据 (motion vector x)
    g_str = exr.channel('G', HALF)  # 获取 G 通道数据 (motion vector y)

    # 将字符串数据转换为 NumPy 数组
    r_array = np.frombuffer(r_str, dtype=np.float16).reshape((height, width))
    g_array = np.frombuffer(g_str, dtype=np.float16).reshape((height, width))

    # 堆叠成 (H, W, 2) 的 shape
    motion_vector = np.stack((g_array, r_array), axis=-1)  # y (G) 在前，x (R) 在后

    return motion_vector

def load_reference(ref_path, device='cpu'):
    # 打开 EXR 文件
    exr = OpenEXR.InputFile(ref_path,)

    # 获取图像尺寸
    dw = exr.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # 读取 R, G, B, A 通道
    HALF = Imath.PixelType(Imath.PixelType.HALF)
    r_str = exr.channel('R', HALF) 
    g_str = exr.channel('G', HALF)
    b_str = exr.channel('B', HALF) 
    a_str = exr.channel('A', HALF)

    # 将字节数据转换为 NumPy 数组
    r_array = np.frombuffer(r_str, dtype=np.float16).reshape((height, width))
    g_array = np.frombuffer(g_str, dtype=np.float16).reshape((height, width))
    b_array = np.frombuffer(b_str, dtype=np.float16).reshape((height, width))
    a_array = np.frombuffer(a_str, dtype=np.float16).reshape((height, width))

    # 将每个通道堆叠成一个 (H, W, C) 数组
    img_array = np.stack((r_array, g_array, b_array, a_array), axis=-1)

    # 转换为 PyTorch tensor，并增加一个 batch 维度 (1, C, H, W)
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).to(device).half()

    return img_tensor

def save_result(result, result_path):
    # 将结果转换为 NumPy 数组
    result = result.squeeze(0)
    rgb_result = result.permute(1, 2, 0).cpu().numpy()
    pyexr.write(result_path, rgb_result)

if __name__ == "__main__":
    for t in range(BEGIN_FRAME, END_FRAME+1):
        FRAME_t = "frame" + f"{t:04}"
        FRAME_f = "frame" + f"{(t-1):04}" 
        p = "1080\\" if IS_USE_1080 else "540\\"
        motion_vector_path = BASE_DIR + "\\" +  FRAME_t + "\\" + p + "motion.exr"        #第t帧的motionvector
        img_path = BASE_DIR + "\\" + FRAME_f + "\\" + p + name + ".exr"               #需要warp的图片（第t-1帧） 
        ref_path = BASE_DIR + "\\" + FRAME_t + "\\" + p + name + ".exr"               #第t帧的参考               
        result_path = BASE_DIR + "\\" + FRAME_t + "\\" + p + "result.exr"               #warp后的结果
        # 加载motion vector
        motion_vector = load_motion_vector(motion_vector_path)
        # 加载reference、img
        img = load_reference(img_path,device='cuda')
        result, grid = warp(img, motion_vector)
        ref = load_reference(ref_path,device='cuda')
        # 保存result
        save_result(result, result_path)
