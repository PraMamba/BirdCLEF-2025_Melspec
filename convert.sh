#!/bin/bash
set -eu

source ~/anaconda3/etc/profile.d/conda.sh
conda activate CZII

MODEL_NAME=cnn_v1
STAGE=train_bce

cd ~/BirdCLEF-2025_Melspec
python3 convert.py --model_name $MODEL_NAME --stage $STAGE --openvino