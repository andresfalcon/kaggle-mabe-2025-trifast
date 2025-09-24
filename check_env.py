import os
import sys
import tomllib
import importlib.util
from pathlib import Path

# =========================
# CONFIG: directorios esperados
# =========================
EXPECTED_DIRS = [
    "src/mabe",
    "configs",
    "work",
    "checkpoints",
    "outputs",
]

EXPECTED_DATA_DIRS = [
    "C:/mabe/data/mabe",
    "C:/mabe/data/mabe/train_annotation",
    "C:/mabe/data/mabe/tracking",
]

EXPECTED_FILES = [
    "configs/config_local.toml",
    "pyproject.toml",
    "C:/mabe/data/mabe/train.csv",
    "C:/mabe/data/mabe/test.csv",
]

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "torch",
    "scipy",
    "sklearn",
    "tomllib",
]

def check_dirs(dirs):
    print("\n📂 Revisando directorios...")
    for d in dirs:
        if Path(d).exists():
            print(f"   ✅ {d}")
        else:
            print(f"   ❌ FALTA: {d}")

def check_files(files):
    print("\n📄 Revisando archivos...")
    for f in files:
        if Path(f).exists():
            print(f"   ✅ {f}")
        else:
            print(f"   ❌ FALTA: {f}")

def check_packages(packages):
    print("\n📦 Revisando librerías instaladas...")
    for pkg in packages:
        if importlib.util.find_spec(pkg) is not None:
            print(f"   ✅ {pkg}")
        else:
            print(f"   ❌ FALTA: {pkg}")

def check_config(config_path="configs/config_local.toml"):
    print("\n⚙️ Revisando configuración...")
    if not Path(config_path).exists():
        print(f"   ❌ FALTA config: {config_path}")
        return
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    paths = cfg.get("paths", {})
    for key, val in paths.items():
        if not Path(val).exists():
            print(f"   ⚠️ {key} apunta a {val} (NO existe)")
        else:
            print(f"   ✅ {key} -> {val}")

def check_env():
    print("\n🔍 Revisando virtual environment...")
    venv_path = os.environ.get("VIRTUAL_ENV") or os.environ.get("UV_PROJECT_ENVIRONMENT")
    if venv_path:
        print(f"   ✅ Usando venv en: {venv_path}")
    else:
        print("   ❌ No se detectó un virtual environment activo.")

if __name__ == "__main__":
    print("=== 🔎 CHECK ENTORNO MABe 2025 ===")
    check_dirs(EXPECTED_DIRS)
    check_dirs(EXPECTED_DATA_DIRS)
    check_files(EXPECTED_FILES)
    check_packages(REQUIRED_PACKAGES)
    check_config()
    check_env()
    print("\n✅ Revisión terminada.")
