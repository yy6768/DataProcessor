"""
Optix SDK/optixDenoiser.exe batch denoise
"""
import subprocess
import os

input_dir = r"G:\optix\input"
output_dir = r"G:\optix\output_motion"
denoiser_path = r"C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0\build\bin\Debug\optixDenoiser.exe"

os.makedirs(output_dir, exist_ok=True)
# 获取所有帧文件夹
frame_folders = [f for f in os.listdir(input_dir) if f.startswith("frame")]

# 遍历每一帧
for frame_folder in frame_folders:
    input_path = os.path.join(input_dir, frame_folder, "color.exr")
    albedo_path = os.path.join(input_dir, frame_folder, "albedo.exr")
    normal_path = os.path.join(input_dir, frame_folder, "normal.exr")
    motion_path = os.path.join(input_dir, frame_folder, "motion.exr")
    output_path = os.path.join(output_dir, f"{frame_folder}.exr")


    # print(f"正在处理: {frame_folder}")
    
    cmd = [
        denoiser_path,
        "-a", albedo_path,
        "-n", normal_path,
        "-f", motion_path,
        "-o", output_path,
        "-k",
        "-z",
        input_path
    ]

    subprocess.run(cmd)
    print(f"Processed: {frame_folder}")
