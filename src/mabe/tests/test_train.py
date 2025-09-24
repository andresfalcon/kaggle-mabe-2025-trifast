import sys
from mabe.utils.config import load_config
from mabe.train import train_main


def main():
    # Config por defecto si no se pasa ruta
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config_local.toml"
    cfg = load_config(cfg_path)

    # Convertimos siempre a dict normal para facilitar los tests
    if hasattr(cfg, "to_dict"):
        cfg = cfg.to_dict()

    # Ajustamos epochs para que el test sea rápido
    cfg["training"]["max_epochs"] = 1

    print("[TEST] Iniciando entrenamiento de prueba con config:", cfg_path)
    train_main(cfg)
    print("[TEST] Entrenamiento de prueba finalizado correctamente.")


if __name__ == "__main__":
    main()
