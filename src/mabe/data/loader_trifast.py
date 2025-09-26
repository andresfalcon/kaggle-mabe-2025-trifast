from __future__ import annotations
import os
import glob
import math
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

# Partes canónicas (orden estable). Si faltan en un video, se omiten.
CANON_PARTS = [
    "body_center", "ear_left", "ear_right", "forepaw_left", "forepaw_right",
    "hindpaw_left", "hindpaw_right", "neck", "nose",
    "tail_base", "tail_midpoint", "tail_tip",
]


def find_tracking_parquet(tracking_root: str, video_id: str) -> Optional[str]:
    """Busca recursivo un parquet que contenga el video_id en el nombre.
    Si no lo encuentra, devuelve el primero (solo para debug)."""
    pats = glob.glob(os.path.join(tracking_root, "**", f"*{video_id}*.parquet"), recursive=True)
    if pats:
        return pats[0]
    pats = glob.glob(os.path.join(tracking_root, "**", "*.parquet"), recursive=True)
    return pats[0] if pats else None


def load_tracking_df(track_path: str) -> pd.DataFrame:
    """Carga parquet y normaliza nombres de columnas: video_frame, mouse_id, bodypart, x, y."""
    df = pq.read_table(track_path).to_pandas()
    lower = {c.lower(): c for c in df.columns}
    rename: Dict[str, str] = {}

    wanted = ["video_frame", "mouse_id", "bodypart", "x", "y"]
    for std in wanted:
        if std not in df.columns:
            cand = None
            for low, orig in lower.items():
                if low.replace(" ", "_") == std:
                    cand = orig
                    break
            if cand is not None:
                rename[cand] = std
    if rename:
        df = df.rename(columns=rename)

    # Tipos
    for c in ["video_frame", "mouse_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # Orden mínimo
    cols_present = [c for c in wanted if c in df.columns]
    if "video_frame" in cols_present and "mouse_id" in cols_present:
        df = df.sort_values(["video_frame", "mouse_id"])
    return df


def egocentricize(xy: np.ndarray, parts: List[str]) -> np.ndarray:
    """Traslada al body_center y rota para alinear (nose - neck) con eje X.
    xy: [T, P, 2]"""
    T, P, _ = xy.shape
    out = xy.copy()

    # Traslación
    if "body_center" in parts:
        idx = parts.index("body_center")
    else:
        idx = 0
    anchor = out[:, idx:idx + 1, :]  # [T,1,2]
    out = out - anchor

    # Rotación por ángulo (nose -> neck)
    if ("nose" in parts) and ("neck" in parts):
        i_nose = parts.index("nose")
        i_neck = parts.index("neck")
        vec = out[:, i_nose, :] - out[:, i_neck, :]  # [T,2]
        ang = np.arctan2(vec[:, 1], vec[:, 0])       # [T]
        ca, sa = np.cos(-ang), np.sin(-ang)
        R = np.stack(
            [
                np.stack([ca, -sa], axis=1),
                np.stack([sa,  ca], axis=1),
            ],
            axis=1,
        )  # [T,2,2]
        out = np.einsum("tpi,tij->tpj", out, R)
    return out


def _sliding_windows(arr: np.ndarray, L: int) -> np.ndarray:
    """Devuelve ventanas deslizantes [W, L, C] con stride 1. Copia segura."""
    if L <= 0:
        return np.zeros((0, 0, arr.shape[1]), dtype=np.float32)
    T = arr.shape[0]
    W = max(T - L + 1, 0)
    if W == 0:
        return np.zeros((0, L, arr.shape[1]), dtype=np.float32)
    # Vista estrideada
    as_strided = np.lib.stride_tricks.as_strided
    strides = (arr.strides[0], arr.strides[0], arr.strides[1])
    shape = (W, L, arr.shape[1])
    win = as_strided(arr, shape=shape, strides=strides)
    return win.copy()


def to_windows(x: np.ndarray, fps: float, short_s: float, mid_s: float, long_s: float
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x: [T, C] → tres tensores [W,L,C] para short/mid/long."""
    Ls = [max(int(round(fps * s)), 1) for s in [short_s, mid_s, long_s]]
    wins = [_sliding_windows(x, L) for L in Ls]
    return tuple(torch.from_numpy(w).float() for w in wins)  # type: ignore


class TriFastFeatureExtractor:
    """Dado un video_id → produce features egocéntricas, derivadas y ventanas tri-ritmo."""

    def __init__(self, train_csv: str, tracking_root: str, windows_cfg: Dict, canon_parts: List[str] = CANON_PARTS):
        self.train_df = pd.read_csv(train_csv)
        self.tracking_root = tracking_root
        self.canon_parts = canon_parts
        self.wcfg = windows_cfg  # dict con short/mid/long (segundos)

    def _fps_for(self, video_id: int) -> float:
        row = self.train_df[self.train_df["video_id"] == video_id]
        if row.empty:
            return 30.0
        cands = ["frames_per_second", "frames per second", "fps", "frame_rate", "FramesPerSecond"]
        for c in cands:
            if c in row.columns:
                try:
                    v = float(row.iloc[0][c])
                    return v if v > 0 else 30.0
                except Exception:
                    continue
        return 30.0

    def load_video(self, video_id: int) -> Dict:
        path = find_tracking_parquet(self.tracking_root, str(video_id))
        if path is None:
            raise FileNotFoundError(f"No encontré parquet para video_id={video_id}")
        df = load_tracking_df(path)

        # Partes canónicas presentes
        parts_present = sorted(
            list(set(df["bodypart"].astype(str)) & set(self.canon_parts)),
            key=self.canon_parts.index,
        )
        if not parts_present:
            raise RuntimeError(f"Sin bodyparts compatibles en {os.path.basename(path)}")

        # Orden por frame/mouse
        df = df.sort_values(["video_frame", "mouse_id"])
        mice = df["mouse_id"].unique().tolist()

        fps = self._fps_for(video_id)
        short_s = float(self.wcfg.get("short", 0.75))
        mid_s = float(self.wcfg.get("mid", 2.5))
        long_s = float(self.wcfg.get("long", 7.5))

        per_mouse = []

        for m in mice:
            d = df[df["mouse_id"] == m]
            # Construimos [T, P, 2]
            T = d["video_frame"].max() + 1 if not d.empty else 0
            P = len(parts_present)
            xy = np.full((T, P, 2), np.nan, dtype=np.float32)

            # Llenar por part
            for p_idx, p in enumerate(parts_present):
                cols_expected = ["video_frame", "x", "y"]
                dp = d[d["bodypart"] == p][cols_expected].dropna()
                if dp.empty:
                    continue
                vf = dp["video_frame"].to_numpy(dtype=int, copy=False)
                xv = dp["x"].to_numpy(dtype=np.float32, copy=False)
                yv = dp["y"].to_numpy(dtype=np.float32, copy=False)
                xy[vf, p_idx, 0] = xv
                xy[vf, p_idx, 1] = yv

            # Interpolación simple por eje tiempo, por part y coord
            for p_idx in range(P):
                for coord in range(2):
                    vec = xy[:, p_idx, coord]
                    nans = np.isnan(vec)
                    if nans.all():
                        xy[:, p_idx, coord] = 0.0
                    else:
                        s = pd.Series(vec)
                        vec2 = s.ffill().bfill().to_numpy(copy=False)
                        xy[:, p_idx, coord] = vec2.astype(np.float32, copy=False)

            # Egocéntrico
            xy = egocentricize(xy, parts_present)  # [T,P,2]

            # Derivadas (velocidad) por diff en tiempo
            vel = np.diff(xy, axis=0, prepend=xy[:1])

            # Apilamos posición+velocidad → [T,P,4]
            feat = np.concatenate([xy, vel], axis=2)
            feat = feat.reshape(feat.shape[0], -1).astype(np.float32)  # [T, C]

            # Normalización robusta por canal (mediana/iqr)
            med = np.nanmedian(feat, axis=0)
            q75 = np.nanpercentile(feat, 75, axis=0)
            q25 = np.nanpercentile(feat, 25, axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = 1.0
            feat = (feat - med) / iqr

            # Tri-ritmo
            short_w, mid_w, long_w = to_windows(feat, fps, short_s, mid_s, long_s)

            per_mouse.append(
                {
                    "mouse_id": int(m),
                    "parts": parts_present,
                    "fps": fps,
                    "short": short_w,
                    "mid": mid_w,
                    "long": long_w,
                }
            )

        return {
            "video_id": int(video_id),
            "fps": fps,
            "parts": parts_present,
            "mice": per_mouse,
            "track_path": path,
        }
