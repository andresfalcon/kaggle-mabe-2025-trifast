from __future__ import annotations
import argparse, sys, torch, os
from torch import nn, optim
from torch.utils.data import DataLoader

from mabe.utils.config import load_config
from mabe.utils.seed import set_seed
from mabe.utils.log import get_logger
from mabe.data.dataset import TriScalePoseDataset
from mabe.models.trifast_pose import TriFastPoseModel


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True,
                    help="Ruta al archivo config TOML")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = load_config(cfg_path)

    # mantenerlo como objeto Cfg (soporta .paths.train_csv, etc.)
    set_seed(cfg.system.seed)
    device = torch.device(cfg.system.device if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Seed fijada en {cfg.system.seed}")
    logger = get_logger("train_trifast", level=cfg.logging.log_level)

    # === Dataset & Loader ===
    train_ds = TriScalePoseDataset(cfg, split="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )

    # === Modelo Trifast ===
    model = TriFastPoseModel(
        input_dim=train_ds[0]["short"].shape[-1],
        hidden_dim=cfg.model.hidden_dim,
        num_classes=len(train_ds[0]["actions_vocab"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # === Entrenamiento ===
    max_epochs = cfg.training.max_epochs
    logger.info(f"Entrenamiento comenzando por {max_epochs} epochs")

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch["short"].to(device)
            y = batch["y_short"].to(device)

            # Flatten
            x = x.view(-1, x.shape[-1])
            y = y.view(-1, y.shape[-1]).argmax(dim=1)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1}/{max_epochs} - Loss: {total_loss:.4f}")

        # Guardar checkpoint
        ckpt_dir = cfg.output.checkpoints_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Checkpoint guardado en {ckpt_path}")

    logger.info("Entrenamiento finalizado.")


if __name__ == "__main__":
    main()
