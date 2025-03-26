import os
import glob
import cv2
import re
import numpy as np
import torch.nn.functional as F
import torch
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from typing import List

scene = "MedievalDocks"
method = "GT"
weight=0.4
sid = 604
eid = 749
rootpath="/disk/zjw/gffe/result/"+scene+"/"+method
mvPath = "/disk/zjw/data/zwtdataset/"+scene+"/test1-60fps/"+scene
imgpath="/disk/zjw/gffe/result/"+scene+"/"+method
respath="/disk/zjw/gffe/result/"+scene+"/"+method+'taa/'

def torchWarp(img1, motion2):
    n, c, h, w = img1.shape

    dx, dy = torch.linspace(-1, 1, w,
                            device=img1.device), torch.linspace(-1, 1, h, device=img1.device)
    grid_y, grid_x = torch.meshgrid(dy, dx, indexing="ij")

    grid_x = grid_x.repeat(n, 1, 1) - (2 * motion2[:, 1] / (w))
    grid_y = grid_y.repeat(n, 1, 1) + (2 * motion2[:, 0] / (h))
    coord = torch.stack([grid_x, grid_y], dim=-1)
    res = F.grid_sample(img1, coord, padding_mode='zeros', align_corners=True)
    return res


def cvWarp(img, mv):
    img = torch.tensor(img.transpose([2, 0, 1])).unsqueeze(0)
    mv = torch.tensor(mv.transpose([2, 0, 1])).unsqueeze(0)
    res = torchWarp(img, mv)
    return res[0].numpy().transpose([1, 2, 0])


def BGR2YCoCg(img):  # img:h,w,3
    res = np.zeros_like(img)
    res[..., 0] = 0.25*img[..., 0]+0.5*img[..., 1]+0.25*img[..., 2]
    res[..., 1] = -0.5*img[..., 0]+0.5*img[..., 2]
    res[..., 2] = -0.25*img[..., 0]+0.5*img[..., 1]-0.25*img[..., 2]
    return res


def YCoCg2BGR(img):  # img:h,w,3
    res = np.zeros_like(img)
    res[..., 0] = img[..., 0]-img[..., 1]-img[..., 2]
    res[..., 1] = img[..., 0]+img[..., 2]
    res[..., 2] = img[..., 0]+img[..., 1]-img[..., 2]
    return res
def TAA_my(weight,respath,exr_list, 
    output_file: str,
    frames_per_image: int = 1,
    video_fps: int = 60) -> list[str]:
    # if not os.path.exists(respath):
    #     os.mkdir(respath)
    ret = []
    j = 0
    print(exr_list[j])
    res = cv2.imread(exr_list[j], cv2.IMREAD_UNCHANGED)[:, :, 0:3].astype(np.float32)#TODO:设置读取的图片文件名
    if 'tonemap' in os.path.basename(exr_list[j]).lower():    
        res = tonemap(res, mode="simple")
    j+=1

    first_frame=(res**(1/2.2)).clip(0,1)
    frame_8u = np.clip(first_frame * 255.0, 0, 255).astype(np.uint8)
    
    height, width = frame_8u.shape[:2]

    # 初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, video_fps, (width, height))

    if not video_writer.isOpened():
        print("无法初始化视频写入器，请检查编码器或权限是否正常。")
        return

    # 写入第一帧（重复 frames_per_image 次）
    for _ in range(frames_per_image):
        video_writer.write(frame_8u)

    # res=BGR2YCoCg(res)
    h,w,c=res.shape
    # 但需要注意taa.py里的weight变量，我设置的是0.4，但不同场景不同情况下应该是需要自行调节的，weight越小时warp造成伪影越多，weight越大锯齿和抖动越厉害
    
    lc=np.array([0.0,-0.5,-0.5]).reshape(1,1,3)
    lc=np.repeat(lc,h,axis=0)
    lc=np.repeat(lc,w,axis=1)
    rc=np.array([1.0,0.5,0.5]).reshape(1,1,3)
    rc=np.repeat(rc,h,axis=0)
    rc=np.repeat(rc,w,axis=1)
    
    
    for i in range(sid+1, eid):
        print(exr_list[j])
        img = cv2.imread(exr_list[j], cv2.IMREAD_UNCHANGED)[:, :, 0:3].astype(np.float32)#TODO:设置读取的图片文件名
        if 'tonemap' in os.path.basename(exr_list[j]).lower():
            print(1)
            img = tonemap(img, mode="simple")
        print(mvPath+"MotionVector.0{}.exr".format(i))
        mv = cv2.imread(mvPath+"MotionVector.0{}.exr".format(i),cv2.IMREAD_UNCHANGED)[:, :, 1:3].astype(np.float32)#TODO:设置读取的motion vector文件名与格式

        warp = cvWarp(res, mv)
        #cv.imwrite(respath+'taaWarp.{}.exr'.format(i), warp)
        h, w, c = warp.shape
        img = BGR2YCoCg(img)  # img=cv.cvtColor(img,cv.COLOR_BGR2YUV)
        warp = BGR2YCoCg(warp)  # warp=cv.cvtColor(warp,cv.COLOR_BGR2YUV)
        unf = np.pad(img, ((1, 1), (1, 1), (0, 0)))
        unf = torch.tensor(unf)
        unf = unf.unfold(0, 3, 1)
        unf = unf.unfold(1, 3, 1)
        unf = unf.reshape(h, w, c, 9).numpy()

        m1=unf.sum(-1)
        m2=np.power(unf,2).sum(-1)
        
        mu=(m1/9)
        sigma=np.sqrt(np.abs(m2/9-mu**2)).clip(0)
        maxn=np.minimum(mu+0.5*sigma,rc).astype(np.float32)
        minn=np.maximum(mu-0.5*sigma,lc).astype(np.float32)
        warp = np.clip(warp, minn, maxn)
        
        res = warp*(1-weight)+img*weight
        res=YCoCg2BGR(res)

        img=(res**(1/2.2)).clip(0,1)
        img_8u = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        # 写入多帧
        for _ in range(frames_per_image):
            video_writer.write(img_8u)

        j+=1
    video_writer.release()
    print(f"视频已生成：{output_file}")
    return  ret     


def tonemap(image, mode="simple"):
    """
    简易版 tonemap 函数，演示对数映射的处理方式。
    注意：这里对 numpy/tensor 做了区分处理，
         如果真实代码中不涉及 torch，可以去掉 torch 相关逻辑。
    """
    if mode == "simple":
        if isinstance(image, np.ndarray):
            # 避免 log(0) 导致 -inf，这里 clip 下限稍微给个正数，比如 1e-8
            return np.log((image + 1).clip(1e-8))
    elif isinstance(image,torch.Tensor):
        if isinstance(image, np.ndarray):
            # 避免 log(0) 导致 -inf，这里 clip 下限稍微给个正数，比如 1e-8
            return torch.log((image + 1).clip(1e-8))
    # 如果需要其他 tonemap 模式，在这里扩展
    return image

def get_exr_files(file_paths: List[str]) -> List[str]:
    """
    根据传入的路径列表，筛选出所有有效的 .exr 文件路径（并进行排序）。
    
    :param file_paths: 可能包含各类文件的路径列表
    :return: 返回按名称排序后的 .exr 文件路径列表
    """
    # 1. 过滤路径是否真实存在
    existing_paths = [p for p in file_paths if os.path.isfile(p)]
    
    # 2. 过滤出结尾是 .exr 的文件
    exr_paths = [p for p in existing_paths if p.lower().endswith('.exr')]

    
    return exr_paths

def exr_sequence_to_mp4(
    exr_files: List[str],
    output_file: str,
    frames_per_image: int = 1,
    video_fps: int = 30
):
    """
    将给定的 EXR 文件路径列表合成为 MP4 视频。

    :param exr_files: EXR 文件路径列表（按顺序）
    :param output_file: 输出的 MP4 文件路径
    :param frames_per_image: 每张图片在视频中占用的帧数
    :param video_fps: 生成视频的帧率 (FPS)
    """

    if not exr_files:
        print("未提供任何 EXR 文件路径，无法生成视频。")
        return

    # 读取第一张 EXR 图片，获取图像分辨率
    first_frame = cv2.imread(exr_files[0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if first_frame is None:
        print(f"无法读取文件 {exr_files[0]}，请检查文件是否有效。")
        return

    # 转换为 8-bit
    if 'tonemap' in os.path.basename(exr_files[0]).lower():
            first_frame = tonemap(first_frame, mode="simple")
    first_frame=(first_frame**(1/2.2)).clip(0,1)
    frame_8u = np.clip(first_frame * 255.0, 0, 255).astype(np.uint8)
    
    height, width = frame_8u.shape[:2]

    # 初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, video_fps, (width, height))

    if not video_writer.isOpened():
        print("无法初始化视频写入器，请检查编码器或权限是否正常。")
        return

    # 写入第一帧（重复 frames_per_image 次）
    for _ in range(frames_per_image):
        video_writer.write(frame_8u)

    # 遍历其余的 EXR 文件
    for exr_file in exr_files[1:]:
        # 读取 EXR 为浮点图像
        img = cv2.imread(exr_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            print(f"警告：无法读取文件 {exr_file}，跳过。")
            continue
        
        # 转换到 8-bit
        if 'tonemap' in os.path.basename(exr_file).lower():
            img = tonemap(img, mode="simple")
        img=(img**(1/2.2)).clip(0,1)
        img_8u = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        # 写入多帧
        for _ in range(frames_per_image):
            video_writer.write(img_8u)

    video_writer.release()
    print(f"视频已生成：{output_file}")


def get_gt_pre_exr(scene="EasternVillage",method="Ours") -> List[str]:
    """
    获取按照“(GT最小序号+6)、PRE最小序号、(GT+8)、PRE下一序号”规律交替排列的 EXR 文件列表。
    当任一侧无法满足继续取文件时，停止拼接并返回最终的路径列表。
    
    :return: 按需求拼接好的 EXR 文件路径列表
    """
    # -------------------------
    # 1) 收集 GT 文件
    # -------------------------
    gt_dir = "/disk/zjw/data/zwtdataset/"+scene+"/test1-60fps"
    # 假设 GT 文件名形如 ****PreTonemapHDRColor.0300.exr
    # 我们用正则找出其中的数字部分
    gt_pattern = re.compile(r"PreTonemapHDRColor\.(\d+)\.exr$", re.IGNORECASE)

    gt_files = glob.glob(os.path.join(gt_dir, "*.exr"))
    gt_index_map = {}  # { index_int: filepath }
    for f in gt_files:
        filename = os.path.basename(f)
        match = gt_pattern.search(filename)
        if match:
            # 转成 int，方便排序和后面查找
            idx = int(match.group(1))
            gt_index_map[idx] = f

    if not gt_index_map:
        print("[警告] 在 GT 文件夹中未找到符合命名规则的 EXR 文件！")
        return []

    # 获取 GT 所有序号并排序
    sorted_gt_indices = sorted(gt_index_map.keys())
    gt_min = sorted_gt_indices[0]  # 最小序号
    gt_max = sorted_gt_indices[-1]
    
    # -------------------------
    # 2) 收集 PRE 文件
    # -------------------------
    pre_dir = "/disk/zjw/gffe/result/"+scene+"/"+method
    # 假设 PRE 文件名形如 pred_color.0000.exr
    pre_pattern = re.compile(r"pred_color\.(\d+)\.exr$", re.IGNORECASE)
    if method == "GT":
        pre_pattern = re.compile(r"label_color\.(\d+)\.exr$", re.IGNORECASE)

    pre_files = glob.glob(os.path.join(pre_dir, "*.exr"))
    pre_index_list = []  # [(index_int, filepath), ...]

    for f in pre_files:
        filename = os.path.basename(f)
        match = pre_pattern.search(filename)
        if match:
            idx = int(match.group(1))
            pre_index_list.append((idx, f))


    if not pre_index_list:
        print("[警告] 在 PRE 文件夹中未找到符合命名规则的 EXR 文件！")
        return []

    # 根据序号排序
    pre_index_list.sort(key=lambda x: x[0])

    # -------------------------
    # 3) 生成拼接顺序
    # -------------------------
    result_paths = []

    # a) 从 GT 的最小序号开始 +6
    current_gt_index = gt_min + 6
    end_index = gt_max - 4

    # b) PRE 从下标 0 开始
    pre_idx = 0

    # 只要能继续找到 GT 或 PRE，就一直拼接
    while True:
        if (current_gt_index not in gt_index_map) or current_gt_index>end_index:
            break

        # 找到了 GT 文件，加入结果列表
        result_paths.append(gt_index_map[current_gt_index])

        # 接下来取 PRE 文件
        if pre_idx >= len(pre_index_list):
            # PRE 文件用完了，停止
            break

        # 拿到当前 PRE 文件
        _, pre_path = pre_index_list[pre_idx]
        result_paths.append(pre_path)
        pre_idx += 1

        # 下一个 GT 序号 = 当前 GT 序号 + 2
        current_gt_index += 2

    return result_paths

def get_gt_30fps_exr(scene="Bunker") -> List[str]:
    """
    从 GT 文件夹中获取 EXR 文件序列，按照“最小序号+6”开始，每次加2，一直到“最大序号-2”结束，
    如果中间某个序号没有对应文件，则跳过。

    :return: 筛选后的 EXR 文件路径列表（有序）
    """

    # GT 文件夹路径
    gt_dir = "/disk/zjw/data/zwtdataset/"+scene+"/test1-60fps"

    # 文件名可能形如 ****PreTonemapHDRColor.0300.exr
    # 用正则匹配末尾的数字序号
    gt_pattern = re.compile(r"PreTonemapHDRColor\.(\d+)\.exr$", re.IGNORECASE)

    # 先收集所有可能的 EXR 文件
    all_exr_files = glob.glob(os.path.join(gt_dir, "*.exr"))

    # 字典：序号 -> 文件路径
    gt_index_map = {}

    for fpath in all_exr_files:
        fname = os.path.basename(fpath)
        match = gt_pattern.search(fname)
        if match:
            # 将序号转换为 int 方便后续排序和查找
            idx = int(match.group(1))
            gt_index_map[idx] = fpath

    # 如果没有匹配到任何 GT 文件，直接返回空列表
    if not gt_index_map:
        print("[警告] 在 GT 文件夹中未找到符合规则的 EXR 文件！")
        return []

    # 获取所有序号并排序
    sorted_gt_indices = sorted(gt_index_map.keys())

    # 最小/最大序号
    gt_min = sorted_gt_indices[0]
    gt_max = sorted_gt_indices[-1]

    # 从 gt_min+6 开始，一直到 gt_max-6，每次 +2
    start_index = gt_min + 6
    end_index = gt_max - 4

    result_paths = []
    # 遍历每个可能的序号
    for current_index in range(start_index, end_index + 1, 2):
        # 如果这个序号存在于 gt_index_map，就存起来
        if current_index in gt_index_map:
            result_paths.append(gt_index_map[current_index])
        # 如果不存在就跳过，不做任何处理

    return result_paths

if __name__ == '__main__':
    # 示例：假设我们有若干个 .exr 文件，分布在不同的目录

    Scenes=["MedievalDocks","Bunker","Factory",
            "EasternVillage","RedwoodForest","Showdown","WesternTown"
            ]
    methods=["GT","LMV","Ours","ExtraNet","Mob-FGSR-E","ExtraSS","GFFE"]

    #for scene in Scenes:
    #exr_list = get_gt_pre_exr(scene=scene,method=method)
    exr_list = get_gt_30fps_exr(scene=scene)
    print(len(exr_list))


    # 获取所有有效的 exr 文件，得到按文件名排序后的列表
    valid_exr_files = get_exr_files(exr_list)

        # # 设置输出视频参数
    output_mp4 = r"./video/taa/"+scene+method+"30FPStaa.mp4"
    frames_per_image = 2    # 每张图重复多少帧
    video_fps = 60          # 最终视频的FPS

    output_dir = os.path.dirname(respath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    TAA_my(weight,respath,valid_exr_files,output_file=output_mp4,
        frames_per_image=frames_per_image,
        video_fps=video_fps)





