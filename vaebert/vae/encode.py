import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from vaebert.vae.data import PartNetVoxelDataset
from vaebert.vae.vae import VAE


def encode_binvoxes(
    model: nn.Module, dataloader: DataLoader, args: argparse.Namespace
) -> None:
    model.eval()

    latents = np.empty((0, args.latent_dim))

    for voxels in tqdm(dataloader):
        voxels = voxels.float().to(args.device)
        with torch.no_grad():
            _, z = model(voxels)
        z = z.cpu().numpy()
        latents = np.concatenate((latents, z))

    dataloader.dataset.shapenet.create_dataset("latents", data=latents)
    dataloader.dataset.shapenet.close()


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
        default=2,
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

    dataset = PartNetVoxelDataset(args.input_path)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    encode_binvoxes(model, dataloader, args)
