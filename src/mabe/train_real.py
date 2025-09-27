from __future__ import annotations
import argparse, sys, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from mabe.utils.config import load_config
from mabe.utils.seed import set_seed
from mabe.utils.log import get_logger
from mabe.data.dataset import TriScalePoseDataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True,
                    help="Ruta al archivo config TOML")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = load_config(cfg_path)

    # Aseguramos dict subscripteable
    if hasattr(cfg, "to_dict"):
        cfg = cfg.to_dict()

    set_seed(cfg["system"]["seed"])
    device = cfg["system"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Seed fijada en {cfg['system']['seed']}")
    logger = get_logger("train_real", level=cfg["logging"]["log_level"])

    # === Dataset & Loader ===
    train_ds = TriScalePoseDataset(cfg, split="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
    )

    input_dim = train_ds[0]["short"].shape[-1]   # última dimensión de features
    num_classes = len(train_ds[0]["y_short"][0]) # tamaño vocab de acciones

    # === Modelo simple inicial ===
    model = nn.Linear(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # === Entrenamiento ===
    max_epochs = cfg["training"]["max_epochs"]
    logger.info(f"Entrenamiento comenzando por {max_epochs} epochs")

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch["short"].to(device)      # (B, T, F)
            y = batch["y_short"].to(device)    # (B, T, C)

            # Flatten: (B*T, F) → (B*T, C)
            x = x.view(-1, x.shape[-1])
            y = y.view(-1, y.shape[-1]).argmax(dim=1)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1}/{max_epochs} - Loss: {total_loss:.4f}")

    logger.info("Entrenamiento finalizado.")


if __name__ == "__main__":
    main()
