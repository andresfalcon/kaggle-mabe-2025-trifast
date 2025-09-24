from __future__ import annotations
from mabe.data.dataset import TriScalePoseDataset
from mabe.utils.config import load_config
import torch

import sys


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_local.toml"
    cfg = load_config(cfg_path)
    ds = TriScalePoseDataset(cfg, split="train")
    item = ds[0]
    A = len(item["actions_vocab"])
    print("A (num acciones):", A)
    print("short X, y:", item["short"].shape, item["y_short"].shape)
    print("mid   X, y:", item["mid"].shape,   item["y_mid"].shape)
    print("long  X, y:", item["long"].shape,  item["y_long"].shape)
    # chequeo básico: dimensiones compatibles
    assert item["short"].shape[0] == item["y_short"].shape[0]
    assert item["mid"].shape[0]   == item["y_mid"].shape[0]
    assert item["long"].shape[0]  == item["y_long"].shape[0]
    # ver si hay al menos alguna etiqueta positiva
    print("positivos short/mid/long:",
          item["y_short"].sum().item(),
          item["y_mid"].sum().item(),
          item["y_long"].sum().item())

if __name__ == "__main__":
    main()
