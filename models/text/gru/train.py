import argparse
import json
import logging
import sys

from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
from transformers import AutoTokenizer, BertModel

from models.text import PartNetTextLatentDataset
from gru_encoder import GRUEncoder


def train(model, bert, tokenizer, train_dataloader, args, logger, optimizer):
    criterion = torch.nn.MSELoss()
    for epoch in tqdm(range(1, args.epochs + 1)):
        hidden_states = [
            torch.zeros(2, args.batch_size, 64).to(args.device),
            torch.zeros(2, args.batch_size, 128).to(args.device),
            torch.zeros(2, args.batch_size, 256).to(args.device),
        ]
        losses = []
        for captions, latents in tqdm(
            train_dataloader,
            total=len(train_dataloader),
            leave=False,
        ):
            optimizer.zero_grad()
            inputs = tokenizer(
                captions, padding=True, truncation=True, return_tensors="pt"
            ).to(args.device)
            bert_embed = bert(**inputs).last_hidden_state
            seq_lengths = inputs["attention_mask"].sum(dim=1).cpu()
            output, hidden_states = model(bert_embed, seq_lengths, hidden_states)
            loss = criterion(output, latents.to(args.device))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        logger.info(f"Epoch: [{epoch}/{args.epochs}], Loss: {np.mean(losses):.3f}")

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), args.output_dir / f"epoch{epoch}.pt")


def collate_fn(batch):
    captions, latents = zip(*batch)
    return list(captions), torch.tensor(latents)


if __name__ == "__main__":
    root_path = Path(__file__).resolve()

    parser = argparse.ArgumentParser(
        description="Training script for VAE on subset of PartNet"
    )
    parser.add_argument(
        "-seed", "--seed", default=488, type=int, help="The seed to put into torch."
    )
    parser.add_argument(
        "-input",
        "--input_path",
        default=root_path.parent.parent.parent.parent / "shapenet",
        type=Path,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "-output",
        "--output_dir",
        default=root_path.parent / "checkpoints",
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
        default=1e-3,
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
        default=1000,
        type=int,
        help="The number of epochs to train the GRUEncoder for.",
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
        default=20,
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
    bert = BertModel.from_pretrained("bert-base-uncased").to(args.device)
    model = GRUEncoder(bert.config.hidden_size, args.latent_dim).to(args.device)

    # Keep the BERT embeddings fixed
    for param in bert.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    logger.info("Starting to train")
    train(model, bert, tokenizer, train_loader, args, logger, optimizer)
