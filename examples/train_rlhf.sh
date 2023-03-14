#!/bin/bash

MODEL="pangu-350M"

#ROOT="/mnt/private-pa002-vol726121-prd/"
ROOT="/root/autodl-tmp/"
DATR_DIR=$ROOT/Data/raw
MAIN=$ROOT/Code/RLHF/src/train_rlhf.py
ACCELERATE_CONFIG=$ROOT/Code/RLHF/src/resources/ppo_model/default_accelerate_config.yaml
MODEL_PATH=$ROOT/Data/models/$MODEL
OUTPUT_DIR=$ROOT/Data/output/rlhf/$MODEL
TRAIN_FILENAME=""
EVAL_FILENAME=""

cd $ROOT/Code/RLHF || exit
mkdir -p $OUTPUT_DIR

#python $MAIN \
accelerate launch --main_process_port 5007 --config_file $ACCELERATE_CONFIG $MAIN \
  --data_dir $DATR_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_PATH \
  --max_length 1024 \
  --logging_steps 100 \
  --do_train \
  --train_filename $TRAIN_FILENAME \
  --train_batch_size 4 \
  --gradient_accumulation_steps 24 \
  --num_epochs 1 \
  --deepspeed_config "ds_config_rlhf_pangu.json" \
  --do_eval \
  --eval_filename $EVAL_FILENAME \
  --eval_batch_size 16 \
  > out/train_rlhf_${MODEL}_"`date "+Y%-%m-%d-%H:%M:%S"`".log 2>&1 &
