import os
from pathlib import Path

# --- Definiciones de paths ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src" / "mabe"
CONFIGS = ROOT / "configs"
DATA = Path("C:/data/mabe")

# --- Estructura esperada ---
expected_files = [
    ROOT / "pyproject.toml",
    ROOT / ".gitignore",
]

expected_dirs = [
    SRC,
    SRC / "data",
    SRC / "models",
    SRC / "tests",
    SRC / "utils",
    CONFIGS,
    DATA,
    DATA / "train_annotation",
    DATA / "train_tracking",
    DATA / "test_tracking",
]

expected_inits = [
    SRC / "__init__.py",
    SRC / "data" / "__init__.py",
    SRC / "models" / "__init__.py",
    SRC / "tests" / "__init__.py",
    SRC / "utils" / "__init__.py",
]

def check_path(p: Path):
    return p.exists()

def main():
    print("=== Verificación de setup Kaggle-MABE-2025 ===")
    ok, missing = [], []

    for f in expected_files:
        (ok if f.exists() else missing).append(str(f))

    for d in expected_dirs:
        (ok if d.exists() else missing).append(str(d))

    for i in expected_inits:
        (ok if i.exists() else missing).append(str(i))

    print("\n✔ Encontrados:")
    for item in ok:
        print("  ", item)

    print("\n❌ Faltantes:")
    for item in missing:
        print("  ", item)

    if not missing:
        print("\n✅ Todo listo!")
    else:
        print("\n⚠️ Revisá los faltantes y crealos antes de avanzar.")

if __name__ == "__main__":
    main()
