import numpy as np
import torch
import torch.nn.functional as F


def compute_motion_vector(motion_vector):
    """
    现在的motion vector H, W, 2, 那么你需要归一化到(-1, 1) 
    TODO: 比如说1080, 1920的画布,现在的某个像素的值是(540, 960), 归一化到(540 / 1080, 960 / 1920)
    """
    pass


def warp(img, motion_vector):
    """
    Warp image using motion vector
    """
    _, _, height, width = img.shape

    # 创建基础网格
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=img.device),
        torch.arange(width, device=img.device),
        indexing="ij",
    )
    # 应用运动向量
    grid_x = grid_x + motion_vector[:, 1] * width
    grid_y = grid_y + motion_vector[:, 0] * height

    # 归一化网格坐标到 [-1, 1] 范围
    grid_x = 2.0 * grid_x / (width - 1) - 1.0
    grid_y = 2.0 * grid_y / (height - 1) - 1.0

    # 堆叠网格坐标
    grid = torch.stack((grid_x, grid_y), dim=1).permute(0, 2, 3, 1)
    # TODO Check the function == cv.remap
    result = F.grid_sample(
        img,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return result, grid

def check_motion_vector(result, next_ref):
    """
    检查motion vector是否正确
    """
    pass

if __name__ == "__main__":
    # 加载motion vector
    motion_vector_path = '/data/hjy/realtimeds_raw/frame0000/540/motion_vector.exr'
    # motion_vector = load_motion_vector(motion_vector_path)
    motion_vector = None
    # 处理motion vector
    compute_motion_vector(motion_vector)
    # 检查motion vector
    # 加载reference
    ref_path = '/data/hjy/realtimeds_raw/frame0000/1080/reference.exr'
    # ref = load_reference(ref_path)
    ref = None
    result, grid = warp(ref, motion_vector)
    next_ref = None
    check_motion_vector(result, next_ref)