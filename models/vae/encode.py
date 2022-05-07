import argparse
import os

import h5py
import numpy as np
import torch

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import PartNetVoxelDataset
from vae import VAE
import json

def encode_binvoxes(model, dataloader, args, dataset_file):
    model.eval()

    latents = np.empty((0, args.latent_dim))

    for voxels in tqdm(dataloader):
        voxels = voxels.float().to(args.device)
        with torch.no_grad():
            _, z = model(voxels)
            print(_)
        z = z.cpu().numpy()
        latents = np.concatenate((latents, z))
    
    dataset_file.create_dataset("latents", data=latents)
    dataset_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Latent vector encoding script for subset of PartNet."
    )
    parser.add_argument(
        "-checkpoint",
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to trained VAE checkpoint.",
    )
    parser.add_argument(
        "-input",
        "--input_path",
        default=os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "shapenet",
            "partnet_data.h5",
        ),
        type=str,
        help="Path to the voxels.",
    )
    parser.add_argument(
        "-output",
        "--output_path",
        default=os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "shapenet",
            "partnet_data.h5",
        ),
        type=str,
        help="The output path to put saved latent vectors.",
    )
    parser.add_argument(
        "-latent",
        "--latent_dim",
        default=128,
        type=int,
        help="The latent dimension size.",
    )
    parser.add_argument(
        "-vox", "--vox_size", default=64, type=int, help="The voxel size."
    )
    parser.add_argument(
        "-batch",
        "--batch_size",
        default=40,
        type=int,
        help="The batch size to use in the dataloader.",
    )
    parser.add_argument(
        "-num_workers",
        "--num_workers",
        default=0,
        type=int,
        help="Number of additional subprocesses loading data.",
    )
    parser.add_argument(
        "-device",
        "--device",
        default="cuda",
        type=str,
        help="Which device to put everything on",
    )
    
    args = parser.parse_args()

    args.device = torch.device(args.device)

    model = VAE(args.latent_dim, args.vox_size).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device))

    dataset_file = h5py.File(args.input_path, "r+")
    
    dataset = PartNetVoxelDataset(dataset_file)

    with open(os.path.join(os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "shapenet",
        ), "test_indexes.json"), "r") as f:
        test_indices = json.load(f)

    dataset = Subset(dataset, test_indices)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    encode_binvoxes(model, dataloader, args, dataset_file)
