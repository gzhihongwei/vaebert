import json
import os

import h5py
import numpy as np

from torch.utils.data import Dataset


class PartNetVoxelDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        super().__init__()

        self.data_path = data_path

        with open(os.path.join(self.data_path, "binvox_metadata.json"), "r") as f:
            self.length = 500  # len(json.load(f)["cat_ids"])

    def __getitem__(self, index: int) -> np.ndarray:
        if not hasattr(self, "shapenet"):
            self.shapenet = h5py.File(self.data_path, "r")["shapes"]

        return self.shapenet[index][np.newaxis, ...]

    def __len__(self) -> int:
        return self.length
