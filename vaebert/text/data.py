from typing import Tuple

import h5py
import numpy as np

from torch.utils.data import Dataset


class PartNetTextLatentDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        super().__init__()
        self.data_path = data_path

    def __getitem__(self, index: int) -> Tuple[str, np.ndarray]:
        if not hasattr(self, "shapenet"):
            self.shapenet = h5py.File(self.data_path, "r")

        return (
            self.shapenet["captions"][index].decode(),
            self.shapenet["latents"][index].astype("float32"),
        )

    def __len__(self) -> int:
        if not hasattr(self, "shapenet"):
            self.shapenet = h5py.File(self.data_path, "r")

        return len(self.shapenet)
