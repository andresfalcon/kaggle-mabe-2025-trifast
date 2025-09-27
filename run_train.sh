#!/bin/bash
set -e

if [ "$1" == "debug" ]; then
  CONFIG="/kaggle/working/mabe/configs/config_kaggle_debug.toml"
elif [ "$1" == "real" ]; then
  CONFIG="/kaggle/working/mabe/configs/config_kaggle.toml"
else
  echo "Uso: bash run_train.sh [debug|real]"
  exit 1
fi

echo "Ejecutando entrenamiento con $CONFIG ..."
PYTHONPATH=/kaggle/working/mabe/src:$PYTHONPATH \
python /kaggle/working/mabe/src/mabe/train_real.py --config $CONFIG
