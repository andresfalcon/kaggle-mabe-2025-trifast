from __future__ import annotations
import os
import glob
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def load_annotations_root(ann_root: str) -> Dict[int, pd.DataFrame]:
    """
    Recorre todas las subcarpetas y carga parquet de anotaciones.
    Devuelve dict: video_id -> DataFrame con columnas mínimas:
      ['video_id','agent_id','target_id','action','start_frame','stop_frame']
    """
    paths = glob.glob(os.path.join(ann_root, "**", "*.parquet"), recursive=True)
    by_video: Dict[int, List[pd.DataFrame]] = {}
    for p in paths:
        df = pq.read_table(p).to_pandas()

        # normalizar nombres
        ren: Dict[str, str] = {}
        cols = {c.lower(): c for c in df.columns}
        for want in ["video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"]:
            if want not in df.columns:
                cand = None
                for low, orig in cols.items():
                    if low.replace(" ", "_") == want:
                        cand = orig
                        break
                if cand:
                    ren[cand] = want
        if ren:
            df = df.rename(columns=ren)

        needed = ["agent_id", "target_id", "action", "start_frame", "stop_frame"]
        if not all(c in df.columns for c in needed):
            continue

        # inferir video_id desde el nombre si no está
        if "video_id" not in df.columns:
            m = re.findall(r"(\d+)", os.path.basename(p))
            vid = int(m[0]) if m else -1
            df["video_id"] = vid

        vid_val = int(df["video_id"].iloc[0])
        cols_keep = ["video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"]
        by_video.setdefault(vid_val, []).append(df[cols_keep])

    merged: Dict[int, pd.DataFrame] = {}
    for vid, chunks in by_video.items():
        merged[vid] = pd.concat(chunks, ignore_index=True)
    return merged


def build_actions_vocab(ann_by_video: Dict[int, pd.DataFrame]) -> List[str]:
    acts = set()
    for df in ann_by_video.values():
        if "action" in df.columns:
            acts.update(df["action"].astype(str).unique().tolist())
    return sorted(list(acts))


def labels_for_windows(
    ann_df: pd.DataFrame, agent_id: int, actions_vocab: List[str],
    win_starts: np.ndarray, win_len: int
) -> np.ndarray:
    """
    ann_df: anotaciones del video
    agent_id: agente de interés
    win_starts: [W] t0 de cada ventana
    win_len: L (frames)
    Devuelve Y [W, A] multi-label (0/1).
    """
    num_actions = len(actions_vocab)
    y = np.zeros((len(win_starts), num_actions), dtype=np.float32)
    if ann_df is None or ann_df.empty or len(win_starts) == 0:
        return y

    df = ann_df[ann_df["agent_id"] == agent_id]
    if df.empty:
        return y

    action_to_idx = {a: i for i, a in enumerate(actions_vocab)}
    segs = df[["action", "start_frame", "stop_frame"]].to_numpy()

    t0 = win_starts.astype(np.int64)
    t1 = t0 + int(win_len)

    for act, s0, s1 in segs:
        aidx = action_to_idx.get(str(act))
        if aidx is None:
            continue
        # solape: max(t0, s0) < min(t1, s1+1)
        left = np.maximum(t0, int(s0))
        right = np.minimum(t1, int(s1) + 1)
        overlap = (right - left) > 0
        if overlap.any():
            y[overlap, aidx] = 1.0

    return y
