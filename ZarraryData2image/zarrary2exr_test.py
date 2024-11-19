import zarr
import numpy as np
import OpenEXR, Imath
import zipfile
import os
from tqdm import tqdm


def zarrary_to_exr_color(zarr_file, exr_folder):
    if not os.path.exists(exr_folder):
        os.makedirs(exr_folder)

    z = zarr.open(zarr_file, mode="r")
    data = np.array(z)
    averaged_data = np.mean(data, axis=-1)

    header = OpenEXR.Header(data.shape[2], data.shape[1])
    header["channels"] = {
        "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "A": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
    }

    # for color
    exr_file_i = os.path.join(exr_folder, f"color.exr")
    exr = OpenEXR.OutputFile(exr_file_i, header)

    R = (averaged_data[0, :, :].astype(np.float32) / 255.0).tobytes()
    G = (averaged_data[1, :, :].astype(np.float32) / 255.0).tobytes()
    B = (averaged_data[2, :, :].astype(np.float32) / 255.0).tobytes()
    A = (averaged_data[3, :, :].astype(np.float32) / 255.0).tobytes()

    exr.writePixels({"R": R, "G": G, "B": B})
    exr.close()
    print(f"{zarr_file} to {exr_file_i} done")


def zarrary_to_exr_reference(zarr_file, exr_folder):
    if not os.path.exists(exr_folder):
        os.makedirs(exr_folder)

    z = zarr.open(zarr_file, mode="r")

    data = np.array(z)
    averaged_data = data  # for reference

    header = OpenEXR.Header(data.shape[2], data.shape[1])
    header["channels"] = {
        "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
    }

    exr_file_i = os.path.join(exr_folder, f"reference.exr")
    exr = OpenEXR.OutputFile(exr_file_i, header)

    R = averaged_data[0, :, :].astype(np.float32).tobytes()
    G = averaged_data[1, :, :].astype(np.float32).tobytes()
    B = averaged_data[2, :, :].astype(np.float32).tobytes()

    exr.writePixels({"R": R, "G": G, "B": B})
    exr.close()


def zarrary_to_exr_others(zarr_file, exr_folder, type):
    if not os.path.exists(exr_folder):
        os.makedirs(exr_folder)

    z = zarr.open(zarr_file, mode="r")

    data = np.array(z)
    averaged_data = np.mean(data, axis=-1)

    header = OpenEXR.Header(data.shape[2], data.shape[1])
    header["channels"] = {
        "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
    }

    exr_file_i = os.path.join(exr_folder, f"{type}.exr")
    exr = OpenEXR.OutputFile(exr_file_i, header)

    R = averaged_data[0, :, :].astype(np.float32).tobytes()
    G = averaged_data[1, :, :].astype(np.float32).tobytes()
    B = averaged_data[2, :, :].astype(np.float32).tobytes()

    exr.writePixels({"R": R, "G": G, "B": B})
    exr.close()

def count_files_in_directory(directory):
    items = os.listdir(directory)
    files = [item for item in items if os.path.isfile(os.path.join(directory, item))]
    return len(files)


def unzip_file(zip_file_path, extract_to_dir):
    if not os.path.exists(extract_to_dir):
        os.makedirs(extract_to_dir)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_dir)
        print(f"Files extracted to {extract_to_dir}")


def unzip_all_in_directory(root_dir, output_dir):
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        output_dir0 = os.path.join(output_dir, folder_name)

        frame_num = count_files_in_directory(folder_path)
        for i in range(frame_num):
            file_name = f"frame{i:04d}.zip"
            zip_file_path = os.path.join(folder_path, file_name)
            if os.path.exists(zip_file_path):
                extract_to_dir = os.path.join(
                    output_dir0, file_name[:-4]
                )  # 去掉.zip后缀
                unzip_file(zip_file_path, extract_to_dir)


def zarrary_to_exr_all_in_directory(root_dir, output_dir):
    for folder_name0 in tqdm(os.listdir(root_dir), desc="Processing root directory"):
        folder_path0 = os.path.join(root_dir, folder_name0)
        output_path0 = os.path.join(output_dir, folder_name0)
        for folder_name in tqdm(
            os.listdir(folder_path0), desc=f"Processing {folder_name0}"
        ):
            folder_path = os.path.join(folder_path0, folder_name)
            if os.path.isdir(folder_path):
                output_file = os.path.join(output_path0, folder_name)

                for file_name in tqdm(
                    os.listdir(folder_path), desc=f"Processing {folder_name}"
                ):
                    if file_name == "motion":
                        if not os.path.exists(output_file):
                            os.makedirs(output_file)
                        zarr_file = os.path.join(folder_path, file_name)

                        zarrary_to_exr_others(zarr_file, output_file, "motion")

                    elif file_name == "color":
                        if not os.path.exists(output_file):
                            os.makedirs(output_file)

                        zarr_file = os.path.join(folder_path, file_name)
                        zarrary_to_exr_color(zarr_file, output_file)

                    elif file_name == "reference":
                        if not os.path.exists(output_file):
                            os.makedirs(output_file)

                        zarr_file = os.path.join(folder_path, file_name)
                        zarrary_to_exr_reference(zarr_file, output_file)

                    elif file_name == "diffuse":
                        if not os.path.exists(output_file):
                            os.makedirs(output_file)

                        zarr_file = os.path.join(folder_path, file_name)
                        zarrary_to_exr_others(zarr_file, output_file, "diffuse")

                    elif file_name == "normal":
                        if not os.path.exists(output_file):
                            os.makedirs(output_file)

                        zarr_file = os.path.join(folder_path, file_name)
                        zarrary_to_exr_others(zarr_file, output_file, "normal")

                    elif file_name == "position":
                        if not os.path.exists(output_file):
                            os.makedirs(output_file)

                        zarr_file = os.path.join(folder_path, file_name)
                        zarrary_to_exr_others(zarr_file, output_file, "position")


def zarrary_to_exr_total(root_dir, output_dir, unzip_file):
    unzip_all_in_directory(root_dir, unzip_file)
    zarrary_to_exr_all_in_directory(unzip_file, output_dir)


zarrary_to_exr_total(
    "/data/hjy/balintio/noisebase/data/sampleset_v1/test8", "./output", "./unzip_file"
)
