import os
import cv2
import numpy as np
from pathlib import Path
import shutil

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class SequenceCropper:
    def __init__(self, 
                 source_dir: str,
                 output_dir: str,
                 lr_size: tuple = (128, 128),
                 hr_size: tuple = (256, 256)):
        """
        初始化序列裁剪器
        Args:
            source_dir: 源数据目录
            output_dir: 输出目录
            lr_size: 低分辨率裁剪大小
            hr_size: 高分辨率裁剪大小
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.lr_size = lr_size
        self.hr_size = hr_size
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置OpenCV读取EXR
        cv2.setNumThreads(8)
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        
    def load_exr(self, path: Path) -> np.ndarray:
        """加载EXR文件"""
        try:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to load {path}")
            return img.astype(np.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def save_exr(self, img: np.ndarray, path: Path):
        """保存EXR文件"""
        try:
            cv2.imwrite(str(path), img.astype(np.float32))
        except Exception as e:
            print(f"Error saving {path}: {e}")

    def crop_image(self, img: np.ndarray, x: int, y: int, size: tuple) -> np.ndarray:
        """裁剪图像"""
        return img[y:y+size[1], x:x+size[0]]

    def process_frame(self, frame_dir: Path, frame_idx: int, num_frames: int):
        """处理单个帧
        Args:
            frame_dir: 帧目录
            frame_idx: 当前帧索引
            num_frames: 总帧数，用于计算连续的frame ID
        """
        # 获取所有需要处理的文件
        lr_dir = frame_dir / "540"
        hr_dir = frame_dir / "1080"
        
        # 读取参考图像确定尺寸
        reference = self.load_exr(hr_dir / "reference.exr")
        if reference is None:
            return
        
        h, w = reference.shape[:2]
        lr_h, lr_w = h//2, w//2  # 540p的尺寸
        
        # 计算可以裁剪出多少个块
        num_crops_x = lr_w // self.lr_size[0]
        num_crops_y = lr_h // self.lr_size[1]
        
        # 获取所有需要处理的LR文件
        lr_files = list(lr_dir.glob("*.exr"))
        
        # 对每个可能的裁剪位置进行处理
        for y in range(num_crops_y):
            for x in range(num_crops_x):
                # 计算偏移量
                lr_x = x * self.lr_size[0]
                lr_y = y * self.lr_size[1]
                hr_x = lr_x * 2
                hr_y = lr_y * 2
                
                # 计算连续的frame ID
                sequence_idx = y * num_crops_x + x
                continuous_frame_id = sequence_idx * num_frames + frame_idx
                
                # 创建输出目录
                out_frame_dir = self.output_dir / f"frame{continuous_frame_id:04d}"
                out_lr_dir = out_frame_dir / "540"
                out_hr_dir = out_frame_dir / "1080"
                
                out_lr_dir.mkdir(parents=True, exist_ok=True)
                out_hr_dir.mkdir(parents=True, exist_ok=True)
                
                # 处理所有LR文件
                for lr_file in lr_files:
                    lr_img = self.load_exr(lr_file)
                    if lr_img is not None:
                        cropped_lr = self.crop_image(lr_img, lr_x, lr_y, self.lr_size)
                        self.save_exr(cropped_lr, out_lr_dir / lr_file.name)
                
                # 处理HR参考图像
                cropped_hr = self.crop_image(reference, hr_x, hr_y, self.hr_size)
                self.save_exr(cropped_hr, out_hr_dir / "reference.exr")
                
                # 保存offset信息
                with open(out_frame_dir / "offset.txt", "w") as f:
                    f.write(f"{lr_x},{lr_y}")
                
                # 保存frame_id信息
                with open(out_frame_dir / "frameId.txt", "w") as f:
                    f.write(str(frame_idx))

    def process_sequence(self):
        """处理整个序列"""
        # 获取所有帧目录
        frame_dirs = sorted([d for d in self.source_dir.iterdir() if d.is_dir()])
        num_frames = len(frame_dirs)
        
        # 处理每一帧
        for frame_idx, frame_dir in enumerate(frame_dirs):
            print(f"Processing frame {frame_idx}: {frame_dir}")
            self.process_frame(frame_dir, frame_idx, num_frames)

def main():
    # 配置参数
    source_dir = "/data/hjy/realtimeds_raw/SponzM/"
    output_dir = "/data/hjy/realtimeds_cropped/Sponza/"
    lr_size = (128, 128)
    hr_size = (256, 256)
    
    # 创建裁剪器并处理序列
    cropper = SequenceCropper(source_dir, output_dir, lr_size, hr_size)
    cropper.process_sequence()

if __name__ == "__main__":
    main()