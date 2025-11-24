import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
from tqdm import tqdm


class TunnelDataset(Dataset):
    def __init__(self, root_dir, cache_to_ram=True):
        """
        Args:
            root_dir (string): Path to '50_TUNNELS_4000_PER_TUNNEL'
            cache_to_ram (bool): If True, loads all data into memory for speed.
        """
        self.file_list = glob.glob(os.path.join(
            root_dir, "env_*", "data", "*.npz"))
        self.cache_to_ram = cache_to_ram
        self.data_cache = []

        print(f"Found {len(self.file_list)} samples.")

        if self.cache_to_ram:
            print("Loading dataset into RAM... (This might take a minute)")
            for fpath in tqdm(self.file_list):
                self.data_cache.append(self.load_file(fpath))
            print("Dataset loaded! Training will be fast.")

    def load_file(self, file_path):
        # Load the .npz file
        data = np.load(file_path)

        # 1. Get Image: (60, 80, 3) -> Need (3, 60, 80) for PyTorch
        img = data['image']
        # Normalize to 0-1 range (assuming raw is 0-255) and swap axes
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        # 2. Get Label: Scalar
        label_val = data['label'].item()

        return torch.tensor(img), torch.tensor([label_val], dtype=torch.float32)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.cache_to_ram:
            return self.data_cache[idx]
        else:
            return self.load_file(self.file_list[idx])
