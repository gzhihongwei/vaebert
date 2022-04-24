import h5py
import numpy as np

from pathlib import Path

from torch.utils.data import Dataset

class PartNetVoxelDataset(Dataset):
    def __init__(self, data_path: Path):
        super().__init__()
        
        self.data_path = data_path
        
    def __getitem__(self, index: int) -> np.ndarray:
        pass
        
        