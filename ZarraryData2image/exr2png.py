import os
import OpenEXR
import Imath
import numpy as np
import imageio
from tqdm import tqdm


def exr_to_png(exr_path, png_path):
    exr_file = OpenEXR.InputFile(exr_path)

    # 获取图像尺寸
    dw = exr_file.header()["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # 读取RGBA通道
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    redstr = exr_file.channel("R", FLOAT)
    greenstr = exr_file.channel("G", FLOAT)
    bluestr = exr_file.channel("B", FLOAT)
    # alphastr = exr_file.channel('A', FLOAT)

    # 将通道数据转换为numpy数组
    red = np.frombuffer(redstr, dtype=np.float32).reshape((height, width))
    green = np.frombuffer(greenstr, dtype=np.float32).reshape((height, width))
    blue = np.frombuffer(bluestr, dtype=np.float32).reshape((height, width))
    # alpha = np.frombuffer(alphastr, dtype=np.float32).reshape((height, width))

    # 合并通道
    img = np.stack([red, green, blue], axis=-1)

    # 归一化到0-255并转换为uint8
    img = (img * 255).astype(np.uint8)

    # 保存为PNG
    imageio.imwrite(png_path, img)

def convert_all_exr_to_png(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    scenes = os.listdir(input_directory)
    for scene in tqdm(scenes, desc="Processing scenes"):
        scene_path = os.path.join(input_directory, scene)
        frames = os.listdir(scene_path)
        for frame in tqdm(frames, desc=f"Processing frames in {scene}", leave=False):
            frame_path = os.path.join(scene_path, frame)
            filenames = os.listdir(frame_path)
            for filename in tqdm(
                filenames, desc=f"Processing files in {frame}", leave=False
            ):
                if filename == "color.exr":
                    exr_path = os.path.join(frame_path, filename)

                    output_scene = os.path.join(output_directory, f"{scene}png")
                    if not os.path.exists(output_scene):
                        os.makedirs(output_scene)
                    output_color = os.path.join(output_scene, "color")
                    if not os.path.exists(output_color):
                        os.makedirs(output_color)
                    png_path = os.path.join(output_color, f"{frame}.png")

                    exr_to_png(exr_path, png_path)
                    
                elif filename == "normal.exr":
                    exr_path = os.path.join(frame_path, filename)

                    output_scene = os.path.join(output_directory, f"{scene}png")
                    if not os.path.exists(output_scene):
                        os.makedirs(output_scene)
                    output_normal = os.path.join(output_scene, "normal")
                    if not os.path.exists(output_normal):
                        os.makedirs(output_normal)
                    png_path = os.path.join(output_normal, f"{frame}.png")

                    exr_to_png(exr_path, png_path)
                    
                elif filename == "diffuse.exr":
                    exr_path = os.path.join(frame_path, filename)

                    output_scene = os.path.join(output_directory, f"{scene}png")
                    if not os.path.exists(output_scene):
                        os.makedirs(output_scene)
                    output_diffuse = os.path.join(output_scene, "difffuse")
                    if not os.path.exists(output_diffuse):
                        os.makedirs(output_diffuse)
                    png_path = os.path.join(output_diffuse, f"{frame}.png")

                    exr_to_png(exr_path, png_path)
                    
                elif filename == "motion.exr":
                    exr_path = os.path.join(frame_path, filename)

                    output_scene = os.path.join(output_directory, f"{scene}png")
                    if not os.path.exists(output_scene):
                        os.makedirs(output_scene)
                    output_motion = os.path.join(output_scene, "motion")
                    if not os.path.exists(output_motion):
                        os.makedirs(output_motion)
                    png_path = os.path.join(output_motion, f"{frame}.png")

                    exr_to_png(exr_path, png_path)
                    
                elif filename == "reference.exr":
                    exr_path = os.path.join(frame_path, filename)

                    output_scene = os.path.join(output_directory, f"{scene}png")
                    if not os.path.exists(output_scene):
                        os.makedirs(output_scene)
                    output_ref = os.path.join(output_scene, "reference")
                    if not os.path.exists(output_ref):
                        os.makedirs(output_ref)
                    png_path = os.path.join(output_ref, f"{frame}.png")

                    exr_to_png(exr_path, png_path)
                    


if __name__ == "__main__":
    input_directory = "../exrset_test/output"
    output_directory = "../output_png"
    convert_all_exr_to_png(input_directory, output_directory)
