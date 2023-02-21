#!/bin/bash

MODEL="pangu-2.6B"

#ROOT="/Users/zeyesun/Documents/"
ROOT="/root/autodl-tmp/"
DATR_DIR=$ROOT/Data/raw
MAIN=$ROOT/Code/RLHF/src/train_reward.py
MODEL_PATH=$ROOT/Data/models/$MODEL
OUTPUT_DIR=$ROOT/Data/reward/output/$MODEL
TRAIN_FILENAME=""
EVAL_FILENAME=""

cd $ROOT/Code/RLHF || exit
mkdir -p $OUTPUT_DIR

deepspeed --num_gpus 1 $MAIN \
  --data_dir $DATR_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_PATH \
  --max_length 1024 \
  --logging_steps 100 \
  --do_train \
  --train_filename $TRAIN_FILENAME \
  --train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_epochs 1 \
  --deepspeed_config "ds_config_reward_pangu.json" \
  --do_eval \
  --eval_filename $EVAL_FILENAME \
  --eval_batch_size 16 \
  > train_reward_${MODEL}_"`date "+Y%-%m-%d-%H:%M:%S"`".log 2>&1 &
