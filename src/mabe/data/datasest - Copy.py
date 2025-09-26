from __future__ import annotations
import torch
from torch.utils.data import Dataset

class TriScalePoseDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg; self.split = split
        self.items = [0]  # placeholder

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        return {"short": torch.zeros(1), "mid": torch.zeros(1), "long": torch.zeros(1), "y": torch.tensor(0)}
