from __future__ import annotations
import numpy as np
import pandas as pd, torch
from torch.utils.data import Dataset
from mabe.data.loader_trifast import TriFastFeatureExtractor
from mabe.data.labels import load_annotations_root, build_actions_vocab, labels_for_windows

class TriScalePoseDataset(Dataset):
    """
    Devuelve:
      item = {
        "video_id", "mouse_id", "fps",
        "short": [W_s, L_s, C], "mid": [W_m, L_m, C], "long": [W_l, L_l, C],
        "y_short": [W_s, A], "y_mid": [W_m, A], "y_long": [W_l, A],
        "actions_vocab": [A] (lista)
      }
    """
    def __init__(self, cfg, split="train"):
        self.cfg = cfg; self.split = split
        paths = cfg.paths
        self.train_csv = paths["train_csv"]
        self.tracking_root = paths["tracking_dir"]
        self.ann_root = paths["ann_dir"]
        self.windows = cfg.windows

        self.df = pd.read_csv(self.train_csv)
        self.fx = TriFastFeatureExtractor(self.train_csv, self.tracking_root, self.windows)

        # Cargar anotaciones por video y vocab global
        self.ann_by_video = load_annotations_root(self.ann_root)
        self.actions_vocab = build_actions_vocab(self.ann_by_video)

        self.video_ids = self.df["video_id"].tolist()

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = int(self.video_ids[idx])
        bundle = self.fx.load_video(vid)
        ann_df = self.ann_by_video.get(vid, pd.DataFrame(columns=["video_id","agent_id","target_id","action","start_frame","stop_frame"]))

        m0 = bundle["mice"][0]  # primer mouse para smoke test
        fps = bundle["fps"]

        # Para etiquetar por ventanas, necesitamos los "t0" de cada ventana
        # Nuestro extractor usa ventanas deslizantes 1-step → t0 = [0..W-1]
        def starts_of(tensor_win):
            # tensor_win shape: [W, L, C]
            W, L = tensor_win.shape[0], tensor_win.shape[1]
            return np.arange(W, dtype=np.int32), int(L)

        y_short = y_mid = y_long = torch.zeros(0, len(self.actions_vocab))
        for k in ["short","mid","long"]:
            wins = m0[k]
            if wins.numel() == 0:
                continue
            t0, L = starts_of(wins)
            y = labels_for_windows(ann_df, m0["mouse_id"], self.actions_vocab, t0, L)
            if k == "short": y_short = torch.from_numpy(y)
            if k == "mid":   y_mid   = torch.from_numpy(y)
            if k == "long":  y_long  = torch.from_numpy(y)

        return {
            "video_id": vid,
            "mouse_id": m0["mouse_id"],
            "fps": fps,
            "short": m0["short"], "mid": m0["mid"], "long": m0["long"],
            "y_short": y_short, "y_mid": y_mid, "y_long": y_long,
            "actions_vocab": self.actions_vocab,
        }
