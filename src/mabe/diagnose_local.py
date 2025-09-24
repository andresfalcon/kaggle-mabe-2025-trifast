from __future__ import annotations
import pandas as pd, glob, os
import pyarrow.parquet as pq

# ====== RUTAS LOCALES
TRAIN_CSV = "C:/data/mabe/train.csv"
TRACK_DIR = "C:/data/mabe/train_tracking"
ANN_DIR   = "C:/data/mabe/train_annotation"

def first_present(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

print("=== Celda A — Labs, FPS y behaviors por video ===")
train = pd.read_csv(TRAIN_CSV)

cols = train.columns.tolist()
print("Columnas en train.csv:", cols[:50])

lab_col = first_present(cols, ["lab_id", "lab", "LabID"])
vid_col = first_present(cols, ["video_id", "video", "VideoID"])
fps_col = first_present(cols, ["frames_per_second", "frames per second", "fps", "frame_rate", "FramesPerSecond"])
beh_col = first_present(cols, ["behaviors_labeled", "behaviors labeled", "behaviors", "labels"])

to_show = [x for x in [lab_col, vid_col, fps_col] if x is not None]
print(train[to_show].head(10) if to_show else train.head(3))
print("Labs únicos:", train[lab_col].nunique() if lab_col else "N/A")
if lab_col:
    print(train[lab_col].value_counts().head())

def count_behaviors(s: str):
    try:
        # behaviors_labeled viene como lista en string -> contamos items
        return len([x for x in str(s).split('"') if "," in x]) or len([x.strip() for x in str(s).split(",") if x.strip()])
    except Exception:
        return 0

if beh_col:
    train["n_behaviors_labeled"] = train[beh_col].apply(count_behaviors)
    print("Distribución #behaviors por video:")
    print(train["n_behaviors_labeled"].describe())
    print("Ejemplos behaviors labeled:")
    print(train[[vid_col, beh_col]].head(3))
else:
    print("[INFO] No encontré columna de 'behaviors_labeled'. Cols:", cols)

# === Celda B — Bodyparts por lab (muestra) ===
print("\n=== Celda B — Bodyparts por lab (muestra) ===")

# Buscar parquet en TODAS las subcarpetas
all_tracks = glob.glob(os.path.join(TRACK_DIR, "**", "*.parquet"), recursive=True)
print(f"[INFO] Parquets en tracking encontrados: {len(all_tracks)}")
by_lab = {}

if all_tracks and len(train) > 0:
    sample_df = train.sample(n=min(20, len(train)), random_state=42)
    for _, row in sample_df.iterrows():
        vid = str(row[vid_col]) if vid_col else None
        # Intentar matchear por video_id en el nombre del archivo
        candidates = [p for p in all_tracks if vid and vid in os.path.basename(p)]
        if not candidates:
            # si no matchea, probamos por carpeta con el lab
            lab = str(row[lab_col]) if lab_col else None
            if lab:
                candidates = [p for p in all_tracks if os.sep + lab + os.sep in p]
        if not candidates:
            continue

        df = pq.read_table(candidates[0]).to_pandas()
        part_col = first_present(df.columns.tolist(), ["bodypart", "part", "keypoint"])
        if part_col is None:
            print("[WARN] Sin columna de bodypart en", os.path.basename(candidates[0]), "Cols:", df.columns.tolist())
            continue
        labv = row[lab_col] if lab_col else "unknown_lab"
        parts = df[part_col].astype(str).unique().tolist()
        by_lab.setdefault(labv, set()).update(parts)

if by_lab:
    print("Resumen bodyparts por lab (hasta 5 labs):")
    for labv, parts in list(by_lab.items())[:5]:
        parts_sorted = sorted(list(parts))
        print(f'Lab {labv}: {parts_sorted[:25]} (total: {len(parts_sorted)})')
else:
    print("[INFO] No se pudo armar resumen de bodyparts (¿carpeta vacía o columnas distintas?).")

# === Celda C — Anotaciones (segmentos de ejemplo) ===
print("\n=== Celda C — Anotaciones (segmentos de ejemplo) ===")
ann_paths = glob.glob(os.path.join(ANN_DIR, "**", "*.parquet"), recursive=True)
print('Parquets de anotación encontrados:', len(ann_paths))
if ann_paths:
    ann = pq.read_table(ann_paths[0]).to_pandas()
    print('Head anotaciones:')
    print(ann.head(10))
    act_col = first_present(ann.columns.tolist(), ["action", "label", "behavior"])
    if act_col:
        print('Acciones únicas (muestra):', ann[act_col].astype(str).unique()[:40])
    else:
        print("[INFO] No encontré columna de acción. Cols:", ann.columns.tolist())
else:
    print("No se encontraron .parquet en train_annotation/ (revisar extracción del ZIP).")
