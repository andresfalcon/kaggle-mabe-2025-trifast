@echo off
setlocal EnableDelayedExpansion
REM === Ir al proyecto ===
cd /d "H:\My Drive\Academico\Kaggle\kaggle-mabe-2025-trifast" || (
  echo [ERROR] No existe el path.
  pause & exit /b 1
)

REM === Verificar 'uv' ===
where uv >nul 2>&1
if errorlevel 1 (
  echo [INFO] 'uv' no encontrado. Instalando con pip...
  py -m pip install --upgrade pip
  py -m pip install uv
)

echo [INFO] uv version:
uv --version

REM === Resolver deps si no hay lock ===
if not exist "uv.lock" (
  echo [INFO] Resolviendo dependencias con uv...
  uv sync
)

REM === Asegurar PYTHONPATH para src ===
set "PYTHONPATH=%CD%\src"

echo.
echo ========= MABe - TriFast =========
echo [1] Entrenar (train.py)
echo [2] Inferir (infer.py) -> submission.csv
echo [3] Shell aqui (entorno listo)
echo ==================================
set /p choice=Elige 1/2/3: 

if "%choice%"=="1" (
  uv run python -m mabe.train --config config.toml
  goto :eof
) else if "%choice%"=="2" (
  uv run python -m mabe.infer --config config.toml --out_csv submission.csv
  echo [OK] Generado submission.csv
  goto :eof
) else (
  cmd /k
)
