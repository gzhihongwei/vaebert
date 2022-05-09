import argparse
import json
import logging
import sys

from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from vaebert import collate_fn
from vaebert.text import PartNetTextLatentDataset
from vaebert.text.bert import BERTEncoder


def train(
    model,
    tokenizer,
    train_dataloader,
    args,
    logger,
    optimizer,
    scheduler,
):

    if args.checkpoint_epoch > 0:
        print("Starting with checkpoint", args.checkpoint_epoch)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, f"epoch{args.checkpoint_epoch}.pt")))

    criterion = nn.MSELoss()
    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train()
        for i, (captions, latents) in tqdm(
            enumerate(train_dataloader, start=1),
            total=len(train_dataloader),
            leave=False,
        ):
            optimizer.zero_grad()
            inputs = tokenizer(
                captions, padding=True, truncation=True, return_tensors="pt"
            ).to(args.device)
            output = model(**inputs)
            loss = criterion(output, latents.to(args.device))
            logger.info(
                f"Epoch: [{epoch}/{args.epochs}], Batch: [{i}/{len(train_dataloader)}], Loss: {loss.item():.3f}"
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), args.output_dir / f"epoch{epoch + args.checkpoint_epoch}.pt")


if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Training script for Bert Linear model on subset of PartNet"
    )
    parser.add_argument(
        "-seed", "--seed", default=488, type=int, help="The seed to put into torch."
    )
    parser.add_argument(
        "-input",
        "--input_path",
        default=root_path.parent.parent.parent / "shapenet",
        type=Path,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "-output",
        "--output_dir",
        default=root_path / "checkpoints",
        type=Path,
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
        "-lr",
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The learning rate to use.",
    )
    parser.add_argument(
        "-warmup",
        "--warmup_ratio",
        default=0.2,
        type=float,
        help="For how much of training to linearly increase the learning rate.",
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
        "-checkpoint",
        "--checkpoint_epoch",
        default=0,
        type=int,
        help="The checkpoint epoch to start with.",
    )
    parser.add_argument(
        "-batch",
        "--batch_size",
        default=64,
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

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(levelname)s:%(name)s:%(asctime)s: %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)

    logger.info(args)

    args.device = torch.device(args.device)

    logger.info(f"Using device: {args.device}")

    dataset = PartNetTextLatentDataset(args.input_path / "partnet_data.h5")

    with open(args.input_path / "train_indexes.json", "r") as f:
        train_indices = json.load(f)

    train_dataset = Subset(dataset, train_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BERTEncoder(args.latent_dim).to(args.device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    num_training_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )

    logger.info("Starting to train")
    train(model, train_loader, args, logger, optimizer, scheduler)
