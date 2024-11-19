import zarr
import numpy as np
import OpenEXR, Imath
import zipfile
import os
import tdqm 


def zarrary_to_exr_color(zarr_file, exr_folder):
    if not os.path.exists(exr_folder):
        os.makedirs(exr_folder)

    z = zarr.open(zarr_file, mode="r")
    data = np.array(z)
    averaged_data = np.mean(data, axis=-1)

    # print(averaged_data.shape)

    header = OpenEXR.Header(data.shape[2], data.shape[3])
    header["channels"] = {
        "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "A": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
    }

    # for color
    for i in range(averaged_data.shape[0]):
        exr_file_i = os.path.join(exr_folder, f"frame_{i}.exr")
        exr = OpenEXR.OutputFile(exr_file_i, header)

        R = (averaged_data[i, 0, :, :].astype(np.float32) / 255.0).tobytes()
        G = (averaged_data[i, 1, :, :].astype(np.float32) / 255.0).tobytes()
        B = (averaged_data[i, 2, :, :].astype(np.float32) / 255.0).tobytes()
        A = (averaged_data[i, 3, :, :].astype(np.float32) / 255.0).tobytes()

        exr.writePixels({"R": R, "G": G, "B": B})
        exr.close()
        print(f"{zarr_file} to {exr_file_i} done")


def zarrary_to_exr_reference(zarr_file, exr_folder):
    if not os.path.exists(exr_folder):
        os.makedirs(exr_folder)

    z = zarr.open(zarr_file, mode="r")

    data = np.array(z)
    averaged_data = data  # for reference

    # print(averaged_data.shape)

    header = OpenEXR.Header(data.shape[2], data.shape[3])
    header["channels"] = {
        "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
    }

    for i in range(averaged_data.shape[0]):
        exr_file_i = os.path.join(exr_folder, f"frame_{i}.exr")
        exr = OpenEXR.OutputFile(exr_file_i, header)

        R = averaged_data[i, 0, :, :].astype(np.float32).tobytes()
        G = averaged_data[i, 1, :, :].astype(np.float32).tobytes()
        B = averaged_data[i, 2, :, :].astype(np.float32).tobytes()

        exr.writePixels({"R": R, "G": G, "B": B})
        exr.close()
        print(f"{zarr_file} to {exr_file_i} done")


def zarrary_to_exr_others(zarr_file, exr_folder):
    if not os.path.exists(exr_folder):
        os.makedirs(exr_folder)

    z = zarr.open(zarr_file, mode="r")

    data = np.array(z)
    averaged_data = np.mean(data, axis=-1)

    # print(averaged_data.shape)

    header = OpenEXR.Header(data.shape[2], data.shape[3])
    header["channels"] = {
        "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
    }

    for i in range(averaged_data.shape[0]):
        exr_file_i = os.path.join(exr_folder, f"frame_{i}.exr")
        exr = OpenEXR.OutputFile(exr_file_i, header)

        R = averaged_data[i, 0, :, :].astype(np.float32).tobytes()
        G = averaged_data[i, 1, :, :].astype(np.float32).tobytes()
        B = averaged_data[i, 2, :, :].astype(np.float32).tobytes()

        exr.writePixels({"R": R, "G": G, "B": B})
        exr.close()
        print(f"{zarr_file} to {exr_file_i} done")


def unzip_file(zip_file_path, extract_to_dir):
    if not os.path.exists(extract_to_dir):
        os.makedirs(extract_to_dir)

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_dir)
        print(f"Files extracted to {extract_to_dir}")


def unzip_all_in_directory(root_dir, output_dir):
    for i in tdqm(range(0, 1024), desc="Unzipping files"):
        file_name = f"scene{i:04d}.zip"
        zip_file_path = os.path.join(root_dir, file_name)
        if os.path.exists(zip_file_path):
            extract_to_dir = os.path.join(output_dir, file_name[:-4])  # 去掉.zip后缀
            unzip_file(zip_file_path, extract_to_dir)


def zarrary_to_exr_all_in_directory(root_dir, output_dir):
    # 遍历根目录中的所有文件夹
    for folder_name in tdqm(os.listdir(root_dir), decs = "Processing root directory"):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            # 遍历文件夹中的所有文件
            output_file = os.path.join(output_dir, folder_name)

            for file_name in tdqm(os.listdir(folder_path), decs = "Processing scene"):
                if file_name == "color":
                    if not os.path.exists(output_file):
                        os.makedirs(output_file)

                    zarr_file = os.path.join(folder_path, file_name)
                    exr_file = os.path.join(output_file, "color")
                    zarrary_to_exr_color(zarr_file, exr_file)

                elif file_name == "reference":
                    if not os.path.exists(output_file):
                        os.makedirs(output_file)

                    zarr_file = os.path.join(folder_path, file_name)
                    exr_file = os.path.join(output_file, "reference")
                    zarrary_to_exr_reference(zarr_file, exr_file)

                elif file_name == "diffuse":
                    if not os.path.exists(output_file):
                        os.makedirs(output_file)

                    zarr_file = os.path.join(folder_path, file_name)
                    exr_file = os.path.join(output_file, "diffuse")
                    zarrary_to_exr_others(zarr_file, exr_file)

                elif file_name == "normal":
                    if not os.path.exists(output_file):
                        os.makedirs(output_file)

                    zarr_file = os.path.join(folder_path, file_name)
                    exr_file = os.path.join(output_file, "normal")
                    zarrary_to_exr_others(zarr_file, exr_file)

                elif file_name == "position":
                    if not os.path.exists(output_file):
                        os.makedirs(output_file)

                    zarr_file = os.path.join(folder_path, file_name)
                    exr_file = os.path.join(output_file, "position")
                    zarrary_to_exr_others(zarr_file, exr_file)


def zarrary_to_exr_total(root_dir, output_dir, unzip_file):
    unzip_all_in_directory(root_dir, unzip_file)
    zarrary_to_exr_all_in_directory(unzip_file, output_dir)


zarrary_to_exr_total(
    "/data/yy/nppd/data/sampleset_v1/train", "./output", "./unzip_file"
)
