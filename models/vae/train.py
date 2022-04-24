import argparse
from distutils.debug import DEBUG
import logging
import os

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from tqdm import tqdm

from data import PartNetVoxelDataset
from vae import VAE


def train(model, dataloader, args, device, logger, optimizer):
    model.train()

    for epoch in range(args.epochs):
        for i, voxels in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            voxels = voxels.float().to(device)
            loss, _ = model(voxels)
            logger.info(
                f"Epoch: [{epoch + 1}/{args.epochs}], Batch: [{i+1}/{len(dataloader)}] Loss: {loss.item()}"
            )
            loss.backward()
            optimizer.step()

        torch.save(
            model.state_dict(), os.path.join(args.output_dir, f"epoch{epoch + 1}.pt")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script for VAE on subset of PartNet"
    )
    parser.add_argument(
        "-seed", "--seed", default=488, type=int, help="The seed to put into torch."
    )
    parser.add_argument(
        "-input",
        "--input_path",
        default=os.path.join(
            os.path.abspath(__file__), "..", "shapenet", "binvox_data.hdf5"
        ),
        type=str,
        help="Path to the voxel hdf5.",
    )
    parser.add_argument(
        "-output",
        "--output_dir",
        default=os.path.join(os.path.abspath(__file__), "checkpoints"),
        type=str,
        help="The output directory to put saved checkpoints.",
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
        "-lr",
        "--learning_rate",
        default=6e-4,
        type=float,
        help="The learning rate to use.",
    )
    parser.add_argument(
        "-weight_decay",
        "--weight_decay",
        default=0.001,
        type=float,
        help="The weight decay to apply.",
    )
    parser.add_argument(
        "-epochs",
        "--epochs",
        default=20,
        type=int,
        help="The number of epochs to train the VAE for.",
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
        "--local_rank", default=-1, type=int, help="The local rank to use"
    )
    parser.add_argument(
        "-device",
        "--device",
        default="cuda",
        type=str,
        help="Which device to put everything on",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format="%(asctime)s:%(name)s:%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    device = torch.device(args.device if args.local_rank == -1 else args.local_rank)

    dataset = PartNetVoxelDataset(args.input_path)
    trainloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    model = VAE(args.latent_dim, args.vox_size).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train(model, trainloader, args, device, logger, optimizer)
