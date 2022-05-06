import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader, Subset

from tqdm import tqdm

from data import PartNetVoxelDataset
from vae import VAE


def train(model, train_dataloader, args, logger, optimizer):
    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train()
        for i, voxels in tqdm(
            enumerate(train_dataloader, start=1), total=len(train_indices), leave=False
        ):
            optimizer.zero_grad()
            voxels = voxels.float().to(args.device)
            loss, _ = model(voxels)
            logger.info(
                f"Epoch: [{epoch}/{args.epochs}], Batch: [{i}/{len(train_dataloader)}], Loss: {loss.item():.3f}"
            )
            loss.backward()
            optimizer.step()

        if epoch % args.save_interval == 0:
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, f"epoch{epoch}.pt")
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
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "shapenet",
        ),
        type=str,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "-output",
        "--output_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"),
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
    parser.add_argument(
        "-save_int",
        "--save_interval",
        default=1,
        type=int,
        help="The number of epochs to wait before saving a checkpoint.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        format="%(levelname)s:%(name)s:%(asctime)s: %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)

    logger.info(args)

    args.device = torch.device(args.device)
    logger.info(f"Using device: {args.device}")

    dataset = PartNetVoxelDataset(os.path.join(args.input_path, "binvox.hdf5"))

    with open(os.path.join(args.input_path, "train_indexes.json"), "r") as f:
        train_indices = json.load(f)

    train_dataset = Subset(dataset, train_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    model = VAE(args.latent_dim, args.vox_size).to(args.device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    logger.info("Starting to train")
    train(model, train_loader, args, logger, optimizer)