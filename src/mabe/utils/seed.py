import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Setea la semilla global para reproducibilidad.
    Aplica a Python, NumPy y PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed fijada en {seed}")
