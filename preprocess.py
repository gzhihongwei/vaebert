import argparse
import json
import os
import pickle
import random
import zipfile

from pathlib import Path

import binvox_rw
import h5py
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from scipy import signal


def process_voxel(model, gaussian_kernel):
    vox = model.data.astype(float)

    # Verify the sparsity
    sparsity = vox.mean()
    if sparsity < 0.01:
        return None, sparsity

    if sparsity < 0.4:
        vox_with_kernel = signal.convolve(vox, gaussian_kernel, mode="same")
        vox = np.where(vox_with_kernel > 0.1, 1, 0)

    # Center the voxels
    max_vals_by_axis = [
        np.max(np.max(vox, axis=2), axis=1),
        np.max(np.max(vox, axis=2), axis=0),
        np.max(np.max(vox, axis=1), axis=0),
    ]

    for axis_idx in range(3):
        first_significant_idx = 0
        last_significant_idx = 127

        for idx in range(128):
            if round(max_vals_by_axis[axis_idx][idx]) > 0:
                first_significant_idx = idx
                break

        for idx in reversed(range(128)):
            if round(max_vals_by_axis[axis_idx][idx]) > 0:
                last_significant_idx = idx
                break

        off_center_val = (
            int((128 - (last_significant_idx - first_significant_idx)) / 2)
            - first_significant_idx
        )
        vox = np.roll(vox, off_center_val, axis=axis_idx)

    vox_tensor = torch.tensor(vox).float()
    m = torch.nn.AvgPool3d(2, stride=2)
    vox_tensor = torch.round(
        m(vox_tensor.reshape(1, 128, 128, 128)).reshape(64, 64, 64)
    ).bool()
    return vox_tensor, sparsity


if __name__ == "__main__":
    shapenet_path = Path(__file__).resolve().parent / "shapenet"
    shapenet_path.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Preprocessing script for loading ShapeNet data"
    )
    parser.add_argument(
        "-seed",
        "--seed",
        default=488,
        type=int,
        help="The seed used to shuffle indexes.",
    )
    parser.add_argument(
        "-shapenet",
        "--shapenet_path",
        default=shapenet_path / "ShapeNetCore.v2.zip",
        type=Path,
        help="Path to the shapenet dataset zip.",
    )
    parser.add_argument(
        "-mid2desc",
        "--mid2desc_path",
        default=shapenet_path / "mid2desc.pkl",
        type=Path,
        help="Path to the model id to description pkl file.",
    )
    parser.add_argument(
        "-output",
        "--output_path",
        default=shapenet_path / "partnet_data.h5",
        type=str,
        help="Path to the location of the output dataset file.",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.mid2desc_path, "rb") as f:
        model_captions = pickle.load(f, encoding="utf-8")

    print("Loading data from ShapeNet zip")
    archive = zipfile.ZipFile(args.shapenet_path)

    model_filename_data = []
    for filename in archive.namelist():
        if "solid" in filename:
            split_path = os.path.normpath(filename).split(os.sep)
            model_id = split_path[2]
            if model_id in model_captions.keys():
                cat_id = split_path[1]
                model_filename_data.append((model_id, cat_id, filename))
    print("Loaded ShapeNet zip")

    used_cat_ids = [
        "02818832",
        "02876657",
        "02880940",
        "03001627",
        "03046257",
        "03325088",
        "03593526",
        "03636649",
        "03642806",
        "03797390",
        "04379243",
    ]
    models_info = {}

    for model_id, cat_id, filename in model_filename_data:
        if model_id not in models_info.keys() and cat_id in used_cat_ids:
            models_info[model_id] = (
                model_id,
                cat_id,
                model_captions[model_id],
                filename,
            )
    models_info = sorted(
        list(models_info.values()), key=lambda m: str(m[1]) + str(m[0])
    )
    print(len(models_info), "models found with captions.")

    model_ids, category_ids, captions, filenames = zip(*models_info)

    # https://github.com/starstorms9/shape/blob/ddbacbd0a9897ac6ca78a42a23732c34c13bda1a/utils.py#L303
    x = np.arange(-6, 7, 1)
    y = np.arange(-6, 7, 1)
    z = np.arange(-6, 7, 1)
    xx, yy, zz = np.meshgrid(x, y, z)

    gaussian_kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * 1**2))

    used_model_ids, used_category_ids, used_captions, used_voxels = [], [], [], []
    num_sparsity_under_001 = 0
    num_sparsity_under_04 = 0
    num_sparsity_over_04 = 0

    for i, filename in enumerate(filenames):
        with archive.open(filename) as f:
            current_model = binvox_rw.read_as_3d_array(f)
            vox_tensor, sparsity = process_voxel(current_model, gaussian_kernel)
            if sparsity < 0.01:
                num_sparsity_under_001 += 1
            elif sparsity < 0.4:
                num_sparsity_under_04 += 1
            else:
                num_sparsity_over_04 += 1

            if vox_tensor is not None:
                used_model_ids.append(model_ids[i])
                used_category_ids.append(category_ids[i])
                used_captions.append(captions[i])
                used_voxels.append(vox_tensor.numpy())
            if i == 0:
                print("Processing", len(filenames), "voxels.")
            elif i % 500 == 0:
                print(i, "voxels processed.")

    archive.close()
    print("Processed", len(filenames), "voxels.")
    print(
        num_sparsity_under_001,
        "voxels found and ignored with a sparsity less than 0.01.",
    )
    print(
        num_sparsity_under_04, "voxels found with an original sparsity less than 0.4."
    )
    print(num_sparsity_over_04, "voxels found with an original sparsity over 0.4.")

    print("Saving data")
    used_voxels = np.array(used_voxels)

    with h5py.File(args.output_path, "w") as hf:
        hf.create_dataset("shapes", data=used_voxels)
        np_captions = np.array(
            [str(i).strip() for i in used_captions], dtype=h5py.special_dtype(vlen=str)
        )
        hf.create_dataset("captions", data=np_captions)
        np_category_ids = np.array(
            [str(i).strip() for i in used_category_ids],
            dtype=h5py.special_dtype(vlen=str),
        )
        hf.create_dataset("category_ids", data=np_category_ids)
        np_model_ids = np.array(
            [str(i).strip() for i in used_model_ids], dtype=h5py.special_dtype(vlen=str)
        )
        hf.create_dataset("model_ids", data=np_model_ids)

    train_indexes, test_indexes, _, _ = train_test_split(np.arange(len(np_category_ids)), np_category_ids, test_size=0.1)

    print("Number of train indexes:", len(train_indexes))
    print("Number of test indexes:", len(test_indexes))

    with open(shapenet_path / "train_indexes.json", "w") as f:
        json.dump(train_indexes, f)

    with open(shapenet_path / "test_indexes.json", "w") as f:
        json.dump(test_indexes, f)

    print("Saved data.")
