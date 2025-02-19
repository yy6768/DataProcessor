"""
crop utils
"""
import numpy as np
import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv


###################
# crop color.exr
###################
def load_color(_color_path):
    """
    导入color.exr文件并输出其数据矩阵
    :param _color_path: 文件路径：eg:../color0.exr
    :return: color，一个ndarray矩阵
    """
    try:
        # 导入图片数据
        _img_data = cv.imread(_color_path, cv.IMREAD_UNCHANGED)
        if _img_data is None:
            print(f"Error loading depth image from {_color_path}")
            return None
        return _img_data.astype(np.float32)
    except Exception as _e:
        print(f"An error occurred while loading the color image: {_e}")
        return None


def crop_color(_color, _color_path, crop_weight, crop_height, frame):
    """
    对数据进行处理并返回处理后的矩阵
    :param crop_height: 裁剪高度
    :param crop_weight: 裁剪宽度
    :param _color: ndarray形式的颜色数据,维度应为(height,weight,BGRA)
    :param _color_path: 文件路径：eg:../color0.exr
    """
    try:
        # 根据文件创建对应子文件夹
        dir_path = ''.join(_color_path.split('.')[:-1])
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        # 获取数据的大小
        weight = _color.shape[1]
        height = _color.shape[0]
        index = 0
        for col in range(0, height, crop_height):
            if height - col < crop_height:
                continue
            for row in range(0, weight, crop_weight):
                if weight - row < crop_weight:
                    continue
                # 根据col,row裁剪图片
                cropped_color = _color[col:col + crop_height, row:row + crop_weight, :]
                # 保存图片
                cropped_color_path = os.path.join(dir_path, f'_{index:0>4d}.exr')
                cv.imwrite(cropped_color_path, cropped_color)
                # 写入crop_offset
                cropped_color_offset_path = os.path.join(dir_path, f'_{index:0>4d}.txt')
                with open(cropped_color_offset_path, "w") as file:
                    file.write(f'{col},{row},{frame}')
                index += 1



    except Exception as _e:
        print(f"An error occurred while crop the color image: {_e}")
        return None


def prepro_color_from_file(_color_path, frame):
    """
    从文件对depth数据进行预处理
    :param _color_path:单个文件地址
    :return: 不进行return
    """
    try:
        # 加载color
        _color_data = load_color(_color_path)
        # 对color进行裁剪
        crop_color(_color_data, _color_path, crop_weight, crop_height, frame)
    except Exception as _e:
        print(f"An error occurred while process file: {_e}")


def prepro_depth_from_dir(_root_dir):
    """
    从文件夹对depth数据进行预处理
    :param _root_dir: 根文件夹地址 eg:_root_dir = "D:\\RealtimeDS\\data_reference\\BistroInterior"
    :return: 不进行return
    """
    for subdir, dirs, files in os.walk(_root_dir):
        for file in files:
            if "color" in file:
                _color_path = os.path.join(subdir, file)
                _dirs = subdir.split("\\")
                frame = 0
                for i in _dirs:
                    if "frame" in i:
                        frame = int(i[-4:])
                        break
                prepro_color_from_file(_color_path, frame)


crop_weight = 64
crop_height = 64
color_path = "D:/RealtimeDS/data_reference/BistroInterior"

if __name__ == "__main__":
    """
    做成了脚本形式
    使用如下形式即可工作
    python depth.py <root_dir> <crop_weight> <crop_height>
    """
    if len(sys.argv) < 4:
        try:
            prepro_depth_from_dir(color_path)
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        color_path = sys.argv[1]
        crop_weight = int(sys.argv[2])
        crop_height = int(sys.argv[3])
        try:
            prepro_depth_from_dir(color_path)
        except Exception as e:
            print(f"An error occurred: {e}")
