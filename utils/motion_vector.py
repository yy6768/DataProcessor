import numpy as np
import torch
import torch.nn.functional as F
import OpenEXR
import Imath
import pyexr
import skimage.metrics
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def compute_motion_vector(motion_vector):
    """
    现在的motion vector H, W, 2, 那么你需要归一化到(-1, 1) 
    TODO: 比如说1080, 1920的画布,现在的某个像素的值是(540, 960), 归一化到(540 / 1080, 960 / 1920)
    """
    motion_vector[:,:, 0] /= 1080
    motion_vector[:,:, 1] /= 1920
    return motion_vector

def backproject_pixel_centers(motion, as_grid = False):
    """Decompresses per-sample radiance from RGBE compressed data

    Args:
        motion (tensor, N2HW): Per-sample screen-space motion vectors (in pixels) 
            see `noisebase.projective.motion_vectors`
        crop_offset (tensor, size (2)): offset of random crop (window) from top left corner of camera frame (in pixels)
        prev_crop_offset (tensor, size (2)): offset of random crop (window) in previous frame
        as_grid (bool): torch.grid_sample, with align_corners = False format

    Returns:
        pixel_position (tensor, N2HW): ij indexed pixel coordinates OR
        pixel_position (tensor, NHW2): xy WH position (-1, 1) IF as_grid
    """
    height = motion.shape[2]
    width = motion.shape[3]
    dtype = motion.dtype
    device = motion.device

    pixel_grid = torch.stack(torch.meshgrid(
        torch.arange(0, height, dtype=dtype, device=device),
        torch.arange(0, width, dtype=dtype, device=device),
        indexing='ij'
    ))

    pixel_pos = pixel_grid - motion

    if as_grid:
        # as needed for grid_sample, with align_corners = False
        pixel_pos_xy = torch.permute(torch.flip(pixel_pos, (1,)), (0, 2, 3, 1)) + 0.5
        image_pos = pixel_pos_xy / torch.tensor([width, height], device=device)
        return image_pos * 2 - 1
    else:
        return pixel_pos

def warp(img, motion_vector):
    """
    Warp image using motion vector
    """
    _, _, height, width = img.shape
    grid = backproject_pixel_centers(motion_vector, as_grid=True)

    result = F.grid_sample(
        img,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
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

def load_motion_vector(motion_vector_path, device='cpu'):
    # 打开 EXR 文件
    exr = OpenEXR.InputFile(motion_vector_path)

    # 获取图像尺寸
    dw = exr.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # 读取 R 和 G 通道
    channels = exr.header()['channels']
    r_str = exr.channel('R', channels['R'].type)  # 获取 R 通道数据 (motion vector x)
    g_str = exr.channel('G', channels['G'].type)  # 获取 G 通道数据 (motion vector y)

    # 将字符串数据转换为 NumPy 数组
    rgb_type = np.float16 if channels['R'].type == Imath.PixelType(Imath.PixelType.HALF) else np.float32
    r_array = np.frombuffer(r_str, dtype=rgb_type).reshape((height, width))
    g_array = np.frombuffer(g_str, dtype=rgb_type).reshape((height, width))

    # 堆叠成 (H, W, 2) 的 shape
    motion_vector = np.stack((g_array, r_array), axis=-1)  # y (G) 在前，x (R) 在后
    motion_vector = torch.tensor(motion_vector).permute(2, 0, 1).unsqueeze(0).to(device)
    return motion_vector

def load_reference(ref_path, device='cpu'):
    # 打开 EXR 文件
    exr = OpenEXR.InputFile(ref_path,)

    # 获取图像尺寸
    dw = exr.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # 读取 R, G, B, A 通道
    channels = exr.header()['channels']
    r_str = exr.channel('R', channels['R'].type) 
    g_str = exr.channel('G', channels['G'].type)
    b_str = exr.channel('B', channels['B'].type) 
    a_str = exr.channel('A', channels['A'].type)

    # 将字节数据转换为 NumPy 数组
    rgb_type = np.float16 if channels['R'].type == Imath.PixelType(Imath.PixelType.HALF) else np.float32
    r_array = np.frombuffer(r_str, dtype=rgb_type).reshape((height, width))
    g_array = np.frombuffer(g_str, dtype=rgb_type).reshape((height, width))
    b_array = np.frombuffer(b_str, dtype=rgb_type).reshape((height, width))
    a_array = np.frombuffer(a_str, dtype=rgb_type).reshape((height, width))

    # 将每个通道堆叠成一个 (H, W, C) 数组
    img_array = np.stack((r_array, g_array, b_array, a_array), axis=-1)

    # 转换为 PyTorch tensor，并增加一个 batch 维度 (1, C, H, W)
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).to(device)

    return img_tensor

def save_result(result, result_path):
    # 将结果转换为 NumPy 数组
    result = result.squeeze(0)
    rgb_result = result.permute(1, 2, 0).cpu().numpy()
    pyexr.write(result_path, rgb_result)

if __name__ == "__main__":
    # 加载motion vector
    # motion_vector_path = 'g:\\RealtimeDS\\data\\BistroExterior\\frame0001\\1080\\motion.exr'
    motion_vector_path = '/data/hjy/realtimeds_raw/BistroExterior/frame0002/1080/motion.exr'
    motion_vector = load_motion_vector(motion_vector_path, device='cuda')
    #motion_vector = None
    # 处理motion vector
    motion_vector = compute_motion_vector(motion_vector)
    # 检查motion vector
    # 加载reference
    # ref_path = 'g:\\RealtimeDS\\data\\BistroExterior\\frame0001\\1080\\reference.exr'
    ref_path = '/data/hjy/realtimeds_raw/BistroExterior/frame0001/1080/reference.exr'
    ref = load_reference(ref_path, device='cuda')
    #ref = None
    result, grid = warp(ref, motion_vector)
    #print(result.shape)
    # next_ref_path = 'g:\\RealtimeDS\\data\\BistroExterior\\frame0002\\1080\\reference.exr'
    next_ref_path = '/data/hjy/realtimeds_raw/BistroExterior/frame0002/1080/reference.exr'
    next_ref = load_reference(next_ref_path, device='cuda')
    #check_motion_vector(result, next_ref)
    # 保存result
    result_path = 'result.exr'
    next_ref_path = 'next_ref.exr'
    save_result(result, result_path)
    save_result(next_ref, next_ref_path)
    save_result(motion_vector, 'motion_vector.exr')

