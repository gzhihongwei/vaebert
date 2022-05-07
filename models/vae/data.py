import h5py
import numpy as np

from torch.utils.data import Dataset


class PartNetVoxelDataset(Dataset):
    def __init__(self, h5_file) -> None:
        super().__init__()
        self.shapenet = h5_file["shapes"]

    def __getitem__(self, index: int) -> np.ndarray:
        return self.shapenet[index][np.newaxis, ...]

    def __len__(self) -> int:
        return len(self.shapenet)
