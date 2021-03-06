import h5py
import numpy as np

from torch.utils.data import Dataset


class PartNetVoxelDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        super().__init__()
        self.data_path = data_path

    def __getitem__(self, index: int) -> np.ndarray:
        if not hasattr(self, "shapenet"):
            self.shapenet = h5py.File(self.data_path, "r+")

        return self.shapenet["shapes"][index][np.newaxis, ...]

    def __len__(self) -> int:
        if not hasattr(self, "shapenet"):
            self.shapenet = h5py.File(self.data_path, "r+")

        return len(self.shapenet["shapes"])
