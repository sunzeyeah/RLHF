#!/bin/bash

MODEL="pangu-2.6B"

#ROOT="/mnt/private-pa002-vol726121-prd/"
ROOT="/root/autodl-tmp/"
DATR_DIR=$ROOT/Data/chatgpt/raw
MAIN=$ROOT/Code/RLHF/src/data_prepare.py
MODEL_PATH=$ROOT/Data/models/$MODEL
OUTPUT_DIR=$ROOT/Data/chatgpt/processed

cd $ROOT/Code/RLHF || exit
mkdir -p $OUTPUT_DIR

python $MAIN \
  --data_dir $DATR_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_PATH