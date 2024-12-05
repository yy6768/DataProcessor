"""
depth utils
"""


###################
# Compute log depth
###################

def load_depth(depth_path):
    pass

def log_depth(depth):
    pass


if __name__ == "__main__":
    # 加载depth
    depth_path = '/data/hjy/realtimeds_raw/frame0000/540/depth.exr'
    # depth = log (1 / depth + 1)

    # 保存处理后的depth (新地址)


    # 当确认处理后的depth没有问题后，在每个depth的路径下创建一个新文件'log_depth.exr'
    log_depth_path = depth_path.replace('.exr', '_log.exr')