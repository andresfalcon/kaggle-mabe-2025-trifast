import sys
import time
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mabe.utils.config import load_config
from mabe.data.dataset import TriScalePoseDataset
from mabe.utils.seed import set_seed


def benchmark(cfg_path: str, subset: int = 500, epochs: int = 1):
    print(f"[BENCH] Config: {cfg_path}, subset={subset}, epochs={epochs}")

    # 1. Cargar configuración
    cfg = load_config(cfg_path)
    set_seed(cfg["system"]["seed"])

    # 2. Dataset (subsample)
    full_ds = TriScalePoseDataset(cfg, split="train")
    indices = random.sample(range(len(full_ds)), min(subset, len(full_ds)))
    ds = torch.utils.data.Subset(full_ds, indices)

    # 3. DataLoader
    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["system"]["num_workers"],
    )
    print(f"[BENCH] Subset cargado: {len(ds)} ejemplos → {len(loader)} batches")

    # 4. Modelo dummy
    model = torch.nn.Sequential(
        torch.nn.Linear(48, cfg["model"]["hidden_dim"]),
        torch.nn.ReLU(),
        torch.nn.Linear(cfg["model"]["hidden_dim"], len(full_ds.actions_vocab)),
    )
    model.to(cfg["system"]["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    # 5. Benchmark loop
    start = time.time()
    n_batches = 0
    for epoch in range(epochs):
        for batch in loader:
            # batch["short"] tiene forma [B, T, F], tomamos última dimensión
            x = batch["short"].to(cfg["system"]["device"]).float()
            y = batch["y_short"][:, 0, :].argmax(dim=1).to(cfg["system"]["device"])

            # Flatten temporal para el dummy
            x = x[:, :, :48].reshape(-1, 48)
            y = y.repeat_interleave(x.shape[0] // y.shape[0])

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            n_batches += 1
    end = time.time()

    elapsed = end - start
    avg_batch = elapsed / n_batches
    print(f"[BENCH] Tiempo total: {elapsed:.2f}s ({n_batches} batches)")
    print(f"[BENCH] Promedio por batch: {avg_batch:.4f}s")

    # 6. Extrapolación
    full_batches = len(full_ds) / cfg["training"]["batch_size"]
    est_epoch = full_batches * avg_batch
    print(f"[BENCH] Estimado 1 epoch (dataset completo): {est_epoch/60:.2f} min")

    return elapsed, est_epoch


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config_local.toml"
    subset = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    benchmark(cfg_path, subset=subset, epochs=epochs)
