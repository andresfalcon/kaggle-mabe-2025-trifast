from __future__ import annotations
import argparse, torch
from mabe.utils.config import load_config
from mabe.utils.seed import set_seed
from mabe.utils.log import get_logger
from mabe.data.dataset import TriScalePoseDataset
import torch
import sys
from mabe.utils.config import load_config



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_local.toml")
    return ap.parse_args()


def main():
    # Si no se pasa ruta, usa el config_local
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config_local.toml"
    cfg = load_config(cfg_path)

    # Convertimos a dict si es necesario (para que sea subscriptable)
    if hasattr(cfg, "to_dict"):
        cfg = cfg.to_dict()

    # Reducimos epochs para que el test sea rápido
    cfg["training"]["max_epochs"] = 1

    print("[TEST] Iniciando entrenamiento de prueba con config:", cfg_path)
    train_main(cfg)
    print("[TEST] Entrenamiento de prueba finalizado correctamente.")


def train(cfg):
    """
    Entrenamiento principal del modelo.
    Por ahora es un stub (esqueleto), se puede ampliar más adelante.
    """
    print("[INFO] Entrenamiento iniciado con config:")
    print(cfg)

    # Semilla para reproducibilidad
    set_seed(cfg["system"].get("seed", 42))

    # Placeholder de modelo
    print("[INFO] Modelo inicializado (dummy).")
    print("[INFO] Entrenamiento simulado OK.")
    return True


def train_main(cfg):
    """
    Wrapper usado por los tests para lanzar el entrenamiento.
    """
    return train(cfg)

if __name__ == "__main__":
    main()
