import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
from pathlib import Path
import shutil
import argparse

class SequenceCropper:
    def __init__(self, 
                 args: argparse.Namespace):
        """
        初始化序列裁剪器
        Args:
            source_dir: 源数据目录
            output_dir: 输出目录
            lr_size: 低分辨率裁剪大小
            hr_size: 高分辨率裁剪大小
        """
        self.source_dir = Path(args.source_dir)
        self.output_dir = Path(args.output_dir)
        self.lr_size = args.lr_size
        self.hr_size = args.hr_size
        self.render_height = args.render_height
        self.render_width = args.render_width
        self.num_frames = args.num_frames
        self.num_workers = args.num_workers if hasattr(args, 'num_workers') else os.cpu_count()
        
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

    def process_crop(self, frame_dir: Path, frame_idx: int, num_frames: int, x: int, y: int, sequence_idx: int):
        """处理单个裁剪区域
        Args:
            frame_dir: 帧目录
            frame_idx: 当前帧索引
            num_frames: 总帧数
            x: 裁剪的x坐标
            y: 裁剪的y坐标
            sequence_idx: 序列索引
        """
        # 计算偏移量
        lr_x = x * self.lr_size[0]
        lr_y = y * self.lr_size[1]
        hr_x = lr_x * 2
        hr_y = lr_y * 2
        
        # 计算连续的frame ID
        continuous_frame_id = sequence_idx * num_frames + frame_idx
        
        # 创建输出目录
        out_frame_dir = self.output_dir / f"frame{continuous_frame_id:04d}"
        out_lr_dir = out_frame_dir / f"{self.render_height}"
        out_hr_dir = out_frame_dir / f"{self.render_height * 2}"
        
        out_lr_dir.mkdir(parents=True, exist_ok=True)
        out_hr_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有需要处理的文件
        lr_dir = frame_dir / f"{self.render_height}"
        hr_dir = frame_dir / f"{self.render_height * 2}"
        
        # 读取参考图像
        reference = self.load_exr(hr_dir / "reference.exr")
        if reference is None:
            return
        
        # 处理HR参考图像
        cropped_hr = self.crop_image(reference, hr_x, hr_y, self.hr_size)
        self.save_exr(cropped_hr, out_hr_dir / "reference.exr")
        
        # 处理所有LR文件
        lr_files = list(lr_dir.glob("*.exr"))
        for lr_file in lr_files:
            lr_img = self.load_exr(lr_file)
            if lr_img is not None:
                cropped_lr = self.crop_image(lr_img, lr_x, lr_y, self.lr_size)
                self.save_exr(cropped_lr, out_lr_dir / lr_file.name)
        
        # 保存offset信息
        with open(out_frame_dir / "offset.txt", "w") as f:
            f.write(f"{lr_x},{lr_y}")
        
        # 保存frame_id信息
        with open(out_frame_dir / "frameId.txt", "w") as f:
            f.write(str(frame_idx))

    def process_frame(self, frame_dir: Path, frame_idx: int, num_frames: int):
        """处理单个帧
        Args:
            frame_dir: 帧目录
            frame_idx: 当前帧索引
            num_frames: 总帧数，用于计算连续的frame ID
        """
        # 读取参考图像确定尺寸
        hr_dir = frame_dir / f"{self.render_height * 2}"
        reference = self.load_exr(hr_dir / "reference.exr")
        if reference is None:
            return
        
        h, w = reference.shape[:2]
        lr_h, lr_w = h//2, w//2  # 低分辨率的尺寸
        
        # 计算可以裁剪出多少个块
        num_crops_x = lr_w // self.lr_size[0]
        num_crops_y = lr_h // self.lr_size[1]
        
        # 创建裁剪任务列表
        crop_tasks = []
        for y in range(num_crops_y):
            for x in range(num_crops_x):
                sequence_idx = y * num_crops_x + x
                crop_tasks.append((x, y, sequence_idx))
        
        # 并行处理所有裁剪任务
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(
                    self.process_crop, 
                    frame_dir, 
                    frame_idx, 
                    num_frames, 
                    x, y, 
                    sequence_idx
                ) for x, y, sequence_idx in crop_tasks
            ]
            concurrent.futures.wait(futures)

    def process_sequence(self):
        """处理整个序列"""
        # 获取所有帧目录
        frame_dirs = sorted([d for d in self.source_dir.iterdir() if d.is_dir()])
        if self.num_frames is None:
            num_frames = len(frame_dirs)
        else:
            num_frames = min(self.num_frames, len(frame_dirs))
        
        # 并行处理每一帧
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(
                    self.process_frame, 
                    frame_dir, 
                    frame_idx, 
                    num_frames
                ) for frame_idx, frame_dir in enumerate(frame_dirs[:num_frames])
            ]
            
            # 显示进度
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                print(f"Processed frame {i+1}/{num_frames}")

def main():
    # 配置参数
    args = argparse.ArgumentParser()
    args.add_argument("--source_dir", type=str, default="E:/RealtimeDS/data/SunTemple/")
    args.add_argument("--output_dir", type=str, default="G:/realtimeds_cropped/SunTemple_0413_1")
    args.add_argument("--lr_size", type=tuple, default=(128, 128))
    args.add_argument("--hr_size", type=tuple, default=(256, 256))
    args.add_argument("--render_height", type=int, default=540)
    args.add_argument("--render_width", type=int, default=960)
    args.add_argument("--num_frames", type=int, default=240)
    args = args.parse_args()
    
    # 创建裁剪器并处理序列
    cropper = SequenceCropper(args)
    cropper.process_sequence()

if __name__ == "__main__":
    main()