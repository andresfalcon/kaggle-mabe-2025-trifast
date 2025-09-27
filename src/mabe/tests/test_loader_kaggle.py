from mabe.utils.config import load_config
from mabe.data.dataset import TriScalePoseDataset

def main():
    cfg = load_config("configs/config_kaggle.toml")
    ds = TriScalePoseDataset(cfg, split="train")

    print("Dataset size:", len(ds))
    item = ds[0]
    print("Ejemplo de item:", {k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in item.items()})

if __name__ == "__main__":
    main()
