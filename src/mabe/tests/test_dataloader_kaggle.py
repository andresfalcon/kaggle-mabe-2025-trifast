from mabe.utils.config import load_config
from mabe.data.dataset import TriScalePoseDataset
from torch.utils.data import DataLoader

def main():
    cfg = load_config("configs/config_kaggle.toml")
    ds = TriScalePoseDataset(cfg, split="train")

    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
    )

    batch = next(iter(loader))
    print("Batch keys:", batch.keys())
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {type(v)}")

if __name__ == "__main__":
    main()
