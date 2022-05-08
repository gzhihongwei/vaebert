import argparse
import collections
import json
import logging
import sys

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
from transformers import AutoTokenizer, BertModel

from vaebert import chamfer_dist, plotVox, tsne
from vaebert.data import collate_fn, PartNetTextVoxelDataset
from vaebert.text import BERTEncoder, GRUEncoder
from vaebert.vae import VAE


def test(bert_encoder, bert, gru_encoder, vae, args, test_loader, logger, tokenizer):
    bert_encoder.eval()
    bert.eval()
    gru_encoder.eval()
    vae.eval()

    hidden_states = [
        torch.zeros(2, 1, 64).to(args.device),
        torch.zeros(2, 1, 128).to(args.device),
        torch.zeros(2, 1, 256).to(args.device),
    ]

    metrics = dict(
        bert=collections.defaultdict(list), gru=collections.defaultdict(list)
    )

    for caption, voxel in tqdm(test_loader):
        inputs = tokenizer(
            caption, padding=True, truncation=True, return_tensors="pt"
        ).to(args.device)

        with torch.no_grad():
            bert_output = bert_encoder(**inputs)
            bert_voxel = vae.decode(bert_output, apply_sigmoid=True)

        with torch.no_grad():
            bert_embed = bert(**inputs).last_hidden_state
            seq_lengths = inputs["attention_mask"].sum(dim=1).cpu()
            gru_output, _ = gru_encoder(bert_embed, seq_lengths, hidden_states)
            gru_voxel = vae.decode(gru_output, apply_sigmoid=True)

        metrics["bert"]["chamfer"].append(chamfer_dist(bert_voxel, voxel))
        metrics["gru"]["chamfer"].append(chamfer_dist(gru_voxel, voxel))

    metrics["bert"]["chamfer"] = np.mean(metrics["bert"]["chamfer"])
    metrics["gru"]["chamfer"] = np.mean(metrics["gru"]["chamfer"])

    logger.info(metrics)


def compare_reconstructions(
    bert_encoder, bert, gru_encoder, vae, args, test_df, tokenizer
):
    bert_encoder.eval()
    bert.eval()
    gru_encoder.eval()
    vae.eval()

    limits = (args.vox_size,) * 3

    hidden_states = [
        torch.zeros(2, 1, 64).to(args.device),
        torch.zeros(2, 1, 128).to(args.device),
        torch.zeros(2, 1, 256).to(args.device),
    ]

    for _, row in test_df.iterrows():
        inputs = tokenizer(
            row["captions"], padding=True, truncation=True, return_tensors="pt"
        ).to(args.device)

        with torch.no_grad():
            bert_output = bert_encoder(**inputs)
            bert_voxel = vae.decode(bert_output, apply_sigmoid=True)

        with torch.no_grad():
            bert_embed = bert(**inputs).last_hidden_state
            seq_lengths = inputs["attention_mask"].sum(dim=1).cpu()
            gru_output, _ = gru_encoder(bert_embed, seq_lengths, hidden_states)
            gru_voxel = vae.decode(gru_output, apply_sigmoid=True)

        print(row["captions"])

        plotVox(
            row["voxels"],
            step=1,
            title=f"{row['category']}: {row['model_ids']}",
            threshold=0.5,
            show_axes=False,
            limits=limits,
        )
        plotVox(
            bert_voxel.cpu().numpy(),
            step=1,
            title=f"BERT Encoder Reconstruction",
            threshold=0.5,
            show_axes=False,
            limits=limits,
        )
        plotVox(
            gru_voxel.cpu().numpy(),
            step=1,
            title=f"GRU Encoder Reconstruction",
            threshold=0.5,
            show_axes=False,
            limits=limits,
        )


if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="The inference script for full text to 3D shape generation"
    )
    parser.add_argument(
        "-seed", "--seed", default=488, type=int, help="The seed to put into torch."
    )
    parser.add_argument(
        "-input",
        "--input_path",
        default=root_path / "shapenet",
        type=Path,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "-checkpoints",
        "--checkpoint_paths",
        required=True,
        nargs="+",
        type=str,
        help="The saved checkpoints in the order of BERT, GRU, and finally VAE.",
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
        "-device",
        "--device",
        default=torch.device("cuda"),
        type=torch.device,
        help="Which device to put everything on",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(
        format="%(levelname)s:%(name)s:%(asctime)s: %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)

    logger.info(args)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    bert_encoder = BERTEncoder(args.latent_dim).to(args.device)
    bert_encoder.load_state_dict(
        torch.load(args.checkpoint_paths[0], map_location=args.device)
    )

    bert = BertModel.from_pretrained("bert-base-uncased").to(args.device)
    gru_encoder = GRUEncoder(bert.config.hidden_size, args.latent_dim).to(args.device)
    gru_encoder.load_state_dict(
        torch.load(args.checkpoint_paths[1], map_location=args.device)
    )

    vae = VAE(args.latent_dim, args.vox_size).to(args.device)
    vae.load_state_dict(torch.load(args.checkpoint_paths[2], map_location=args.device))

    dataset = PartNetTextVoxelDataset(args.input_path / "partnet_data.h5")

    with open(args.input_path / "test_indexes.json", "r") as f:
        test_indices = json.load(f)

    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    test(bert_encoder, bert, gru_encoder, vae, test_loader, logger, tokenizer)

    with open(args.input_path / "train_indexes.json", "r") as f:
        train_indices = json.load(f)

    if not hasattr(dataset, "shapenet"):
        dataset.shapenet = h5py.File(dataset.data_path, "r")

    train_latents = dataset.shapenet["latents"][()][train_indices]
    train_voxels = dataset.shapenet["shapes"][()][train_indices]
    train_data = dict(
        model_ids=list(
            map(bytes.decode, dataset.shapenet["model_ids"][()][train_indices])
        ),
        category_ids=list(
            map(bytes.decode, dataset.shapenet["category_ids"][()][train_indices])
        ),
        captions=list(
            map(bytes.decode, dataset.shapenet["captions"][()][train_indices])
        ),
    )

    train_df = pd.DataFrame(train_data)

    category_map = {
        "04379243": "Table",
        "03001627": "Chair",
        "03636649": "Lamp",
        "03325088": "Faucet",
        "03046257": "Clock",
        "02876657": "Bottle",
        "03593526": "Vase",
        "03642806": "Laptop",
        "02818832": "Bed",
        "03797390": "Mug",
        "02880940": "Bowl",
    }

    train_df["Category"] = train_df["category_ids"].map(category_map)

    tsne(train_latents, train_df)

    test_data = dict(
        model_ids=list(
            map(bytes.decode, dataset.shapenet["model_ids"][()][test_indices])
        ),
        category_ids=list(
            map(bytes.decode, dataset.shapenet["category_ids"][()][test_indices])
        ),
        captions=list(
            map(bytes.decode, dataset.shapenet["captions"][()][test_indices])
        ),
        voxels=[
            vox
            for vox in dataset.shapenet["shapes"][()][test_indices].astype(np.float32)
        ],
    )
    test_df = pd.DataFrame(test_data)
    test_df["category"] = test_df["category_ids"].map(category_map)

    sampled_df = test_df.groupby("category_ids").agg(pd.DataFrame.sample)
    compare_reconstructions(
        bert_encoder, bert, gru_encoder, vae, args, sampled_df, tokenizer
    )
