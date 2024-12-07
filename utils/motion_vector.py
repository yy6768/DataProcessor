import numpy as np
import torch
import torch.nn.functional as F
import OpenEXR
import Imath
import imageio
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

    pass


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
    # TODO Check the function == cv.remap（暂时未解决）

    '''map_x = grid_x.cpu().numpy().astype(np.float32)
    map_y = grid_y.cpu().numpy().astype(np.float32)
    # 使用 cv2.remap 进行图像重映射
    result = cv2.remap(img.cpu().numpy(), map_x, map_y, interpolation=cv2.INTER_LINEAR)'''


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

    '''channel_names = exr.header()['channels'].keys()
    
    # 查看每个通道的数据类型
    for channel_name in channel_names:
        pixel_type = exr.header()['channels'][channel_name].type
        if pixel_type == Imath.PixelType(Imath.PixelType.FLOAT):
            print(f"Channel '{channel_name}' is of type FLOAT32.")
        elif pixel_type == Imath.PixelType(Imath.PixelType.HALF):
            print(f"Channel '{channel_name}' is of type FLOAT16.")
        else:
            print(f"Channel '{channel_name}' is of unknown type: {pixel_type}")'''

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
    # 加载motion vector
    motion_vector_path = 'g:\\RealtimeDS\\data\\BistroExterior\\frame0001\\1080\\motion.exr'
    #motion_vector_path = '/data/hjy/realtimeds_raw/frame0000/1080/motion.exr'
    motion_vector = load_motion_vector(motion_vector_path)
    #motion_vector = None
    # 处理motion vector
    compute_motion_vector(motion_vector)
    # 检查motion vector
    # 加载reference
    ref_path = 'g:\\RealtimeDS\\data\\BistroExterior\\frame0001\\1080\\reference.exr'
    #ref_path = '/data/hjy/realtimeds_raw/frame0000/1080/reference.exr'
    ref = load_reference(ref_path,device='cuda')
    #ref = None
    result, grid = warp(ref, motion_vector)
    #print(result.shape)
    next_ref_path = 'g:\\RealtimeDS\\data\\BistroExterior\\frame0002\\1080\\reference.exr'
    #next_ref_path = '/data/hjy/realtimeds_raw/frame0001/1080/reference.exr'
    next_ref = load_reference(next_ref_path,device='cuda')
    #check_motion_vector(result, next_ref)
    # 保存result
    result_path = 'g:\\RealtimeDS\\data\\BistroExterior\\frame0001\\1080\\result.exr'
    save_result(result, result_path)
