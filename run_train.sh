#!/bin/bash
# Script de conveniencia para entrenar en Kaggle

set -e

if [ "$1" == "debug" ]; then
  CONFIG="configs/config_kaggle_debug.toml"
elif [ "$1" == "real" ]; then
  CONFIG="configs/config_kaggle.toml"
else
  echo "Uso: bash run_train.sh [debug|real]"
  exit 1
fi

echo "Ejecutando entrenamiento con $CONFIG ..."
python src/mabe/train_real.py --config $CONFIG
