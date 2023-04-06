#!/bin/bash

MODEL="pangu-2.6B"

ROOT="/mnt/sfevol775196/sunzeye273"
#ROOT="/mnt/share-pa002-vol682688-prd/sunzeye273"
#ROOT="/mnt/pa002-28359-vol543625-private"
#ROOT="/root/autodl-tmp/"
DATR_DIR=$ROOT/Data/chatgpt/raw
#MAIN=$ROOT/Code/chatgpt/src/data_prepare.py
MAIN=$ROOT/Code/RLHF/src/data_prepare.py
MODEL_PATH=$ROOT/Data/models/$MODEL
#MODEL_PATH=/mnt/pa002-28359-vol543625-share/LLM-data/checkpoint/$MODEL
OUTPUT_DIR=$ROOT/Data/chatgpt/processed

#cd $ROOT/Code/chatgpt || exit
cd $ROOT/Code/RLHF || exit
mkdir -p $OUTPUT_DIR

python $MAIN \
  --data_dir $DATR_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_PATH