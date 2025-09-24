import sys
import random
from mabe.utils.config import load_config
from mabe.data.dataset import TriScalePoseDataset
from torch.utils.data import DataLoader

def main():
    if len(sys.argv) < 2:
        print("Uso: uv run python src/mabe/tests/test_dataloader.py configs/config_local.toml")
        sys.exit(1)

    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path)

    print("[TEST] Iniciando DataLoader con config:", cfg_path)

    # Dataset en modo train
    ds = TriScalePoseDataset(cfg, split="train")
    print(f"[INFO] Dataset cargado: {len(ds)} ejemplos totales")

    # Elegimos subset de 10 muestras aleatorias
    indices = random.sample(range(len(ds)), min(10, len(ds)))
    subset = [ds[i] for i in indices]

    print(f"[INFO] Subset creado con {len(subset)} ejemplos")

    # DataLoader pequeño
    loader = DataLoader(subset, batch_size=2, shuffle=True)

    for i, batch in enumerate(loader):
        print(f"\n[INFO] Batch {i+1}:")
        for k, v in batch.items():
            if hasattr(v, "shape"):
                print(f"  {k}: {tuple(v.shape)}")
            else:
                print(f"  {k}: {v}")

    print("\n[TEST] DataLoader subset OK ✅")

if __name__ == "__main__":
    main()
