import zarr
import numpy as np
import os


def raw2zarray(raw_path, zarr_path):
    """
    Input:
    540p:
    - Gbuffer:
        - Albedo
        - Normal
        - Depth
    - Noisy
        - Diffuse
        - Specular
        - Color
    1080p:
    - Reference
    """
    for scene in os.listdir(raw_path):
        pass


if __name__ == "__main__":
    pass