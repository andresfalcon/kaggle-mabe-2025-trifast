from __future__ import annotations
import argparse, torch, pandas as pd
from mabe.utils.config import load_config
from mabe.utils.log import get_logger

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_local.toml")
    ap.add_argument("--out_csv", type=str, default="submission.csv")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Dispositivo: {device}")
    # Escribir un submission.csv de ejemplo con el formato correcto
    sub = pd.DataFrame({
        "row_id":[0],
        "video_id":[101686631],
        "agent_id":["mouse1"],
        "target_id":["mouse2"],
        "action":["sniff"],
        "start_frame":[0],
        "stop_frame":[10]
    })
    sub.to_csv(args.out_csv, index=False)
    logger.info(f"submission.csv escrito: {args.out_csv}")

if __name__ == "__main__":
    main()
