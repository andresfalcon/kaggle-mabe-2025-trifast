from __future__ import annotations
from mabe.data.dataset import TriScalePoseDataset
from mabe.utils.config import load_config

def main():
    cfg = load_config("config_local.toml")
    ds = TriScalePoseDataset(cfg, split="train")
    item = ds[0]
    print("video_id:", item["video_id"], "mouse_id:", item["mouse_id"], "fps:", item["fps"])
    for k in ["short","mid","long"]:
        x = item[k]
        print(k, x.shape)  # [W, L, C]

if __name__ == "__main__":
    main()
