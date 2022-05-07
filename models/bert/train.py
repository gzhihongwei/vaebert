import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import PartNetTextLatentDataset
from bert_encoder import BERTEncoder


def train(
    model,
    tokenizer,
    train_dataloader,
    test_dataloader,
    args,
    logger,
    optimizer,
    scheduler,
):
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

        if epoch % args.test_interval:
            test(model, tokenizer, epoch, test_dataloader, args.device, logger)

        torch.save(
            model.state_dict(), os.path.join(args.output_dir, f"epoch{epoch}.pt")
        )


def test(model, tokenizer, epoch, test_dataloader, device, logger):
    model.eval()
    criterion = nn.MSELoss()
    losses = []

    for captions, latent_vectors in test_dataloader:
        inputs = tokenizer(
            captions, padding=True, truncation=True, return_tensors="pt"
        ).to(args.device)

        with torch.no_grad():
            output = model(**inputs)
            loss = criterion(output, latent_vectors.to(device))
        losses.append(loss.item())

    test_loss = np.mean(losses)

    logger.info(f"Test loss after epoch {epoch}: {test_loss:.3f}")


def collate_fn(batch):
    captions, latents = zip(*batch)
    return list(captions), torch.tensor(latents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script for Bert Linear model on subset of PartNet"
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
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "bert_linear_checkpoints"
        ),
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
    parser.add_argument(
        "-test_int",
        "--test_interval",
        default=2,
        type=int,
        help="The number of epochs to wait before trying the test set",
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

    dataset = PartNetTextLatentDataset(os.path.join(args.input_path, "partnet_data.h5"))

    with open(os.path.join(args.input_path, "train_indexes.json"), "r") as f:
        train_indices = json.load(f)

    with open(os.path.join(args.input_path, "test_indexes.json"), "r") as f:
        test_indices = json.load(f)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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
    train(model, tokenizer, train_loader, test_loader, args, logger, optimizer, scheduler)
