"""
depth utils
"""
import numpy as np
import os
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv


###################
# Compute log depth
###################

def load_depth(_depth_path):
    """
    导入depth.exr文件并输出其数据矩阵
    :param _depth_path: 文件路径：eg:...exr
    :return: depth，一个ndarray矩阵
    """
    try:
        _img_data = cv.imread(_depth_path, cv.IMREAD_UNCHANGED)
        if _img_data is None:
            print(f"Error loading depth image from {_depth_path}")
            return None
        return _img_data.astype(np.float32)
    except Exception as _e:
        print(f"An error occurred while loading the depth image: {_e}")
        return None


def log_depth(_depth):
    """
    对数据进行处理并返回处理后的矩阵
    :param _depth: ndarray形式的深度数据,要求至少为3维且depth存储在BGR中的R值
    :return:一个进行过log处理的深度数据，d=log(1/d+1)
    """
    try:
        # 将depth数据进行切片并处理
        _depth_data = _depth[:, :, 2]
        _depth_data = np.log(1 / (_depth_data+1) + 1)
        _depth[:, :, 2] = _depth_data
        return _depth

    except Exception as _e:
        print(f"An error occurred while log the depth image: {_e}")
        return None


def prepro_depth_from_file(_depth_path):
    """
    从文件对depth数据进行预处理
    :param _depth_path:单个文件地址
    :return: 不进行return
    """
    try:
        # 加载depth
        _depth_data = load_depth(_depth_path)
        # depth = log (1 / (depth+1) + 1)
        _log_depth_data = log_depth(_depth_data)
        # 保存处理后的depth (新地址)
        _log_depth_path = _depth_path.replace('.exr', '_log.exr')
        # 当确认处理后的depth没有问题后，在每个depth的路径下创建一个新文件'log_depth.exr'
        cv.imwrite(_log_depth_path, _log_depth_data)
    except Exception as _e:
        print(f"An error occurred while process file: {_e}")


def prepro_depth_from_dir(_root_dir):
    """
    从文件夹对depth数据进行预处理
    :param _root_dir: 根文件夹地址 eg:_root_dir = "D:\\RealtimeDS\\data\\BistroInterior"
    :return: 不进行return
    """
    for subdir, dirs, files in os.walk(_root_dir):
        for file in files:
            if file == "depth.exr":
                _depth_path = os.path.join(subdir, file)
                prepro_depth_from_file(_depth_path)
            elif file == "montion.exr":
                _montion_path = os.path.join(subdir, file)
                _motion_path = os.path.join(subdir, "motion.exr")
                os.rename(_montion_path, _motion_path)




if __name__ == "__main__":
    """
    做成了脚本形式
    使用如下形式即可工作
    python depth.py <argument1> <argument2>
    """
    if len(sys.argv) < 2:
        print("Usage: python depth.py <argument1> <argument2> ...")
        sys.exit(1)  # Exit with an error code
    for depth_path in sys.argv[1:]:
        try:
            prepro_depth_from_dir(depth_path)
        except Exception as e:
            print(f"An error occurred: {e}")
            continue
