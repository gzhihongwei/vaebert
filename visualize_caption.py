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


def compare_reconstructions(
    bert_encoder, bert, gru_encoder, vae, args, tokenizer
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

    while True:
        caption = input("Enter a caption (optionally add \">\" followed by voxel threshold): ")
        threshold = 0.5
        if len(caption.split(">")) > 1:
            threshold = float(caption.split(">")[1].strip())
            caption = caption.split(">")[0].strip()
        if caption == "":
            break
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

        print(caption)
        
        plotVox(
            bert_voxel.cpu().numpy(),
            step=1,
            title=f"BERT Encoder Reconstruction for \"{caption}\"",
            threshold=threshold,
            show_axes=False,
            limits=limits,
        )
        plotVox(
            gru_voxel.cpu().numpy(),
            step=1,
            title=f"GRU Encoder Reconstruction for \"{caption}\"",
            threshold=threshold,
            show_axes=False,
            limits=limits,
        )


if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Interactve text to shape visualization"
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

    compare_reconstructions(
        bert_encoder, bert, gru_encoder, vae, args, tokenizer
    )
