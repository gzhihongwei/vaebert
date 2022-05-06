import json
import os

import h5py
import numpy as np

from torch.utils.data import Dataset


class PartNetVoxelDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        super().__init__()

        self.data_path = data_path

    def __getitem__(self, index: int) -> np.ndarray:
        if not hasattr(self, "shapenet"):
            self.shapenet = h5py.File(self.data_path, "r")["shapes"]
        return self.shapenet[index][np.newaxis, ...]

    def __len__(self) -> int:
        if not hasattr(self, "shapenet"):
            self.shapenet = h5py.File(self.data_path, "r")["shapes"]

        return len(self.shapenet)


class PartNetCaptionLatentVectorDataset(Dataset):
    def __init__(self, data_path_captions: str, data_path_latents: str) -> None:
        super().__init__()
        self.data_path_captions = data_path_captions
        self.data_path_latents = data_path_latents

    def __getitem__(self, index: int) -> np.ndarray:
        if not hasattr(self, "latents"):
            self.latents = h5py.File(self.data_path_latents, "r")["latents"]
            self.captions = h5py.File(self.data_path_captions, "r")["captions"]
        return self.captions[index].decode(), self.latents[index][np.newaxis, ...]

    def __len__(self) -> int:
        if not hasattr(self, "latents"):
            self.latents = h5py.File(self.data_path_latents, "r")["latents"]
            self.captions = h5py.File(self.data_path_captions, "r")["captions"]

        return len(self.latents)

