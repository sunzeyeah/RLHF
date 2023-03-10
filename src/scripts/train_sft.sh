#!/bin/bash

MODEL="pangu-2.6B"

#ROOT="/mnt/private-pa002-vol726121-prd/"
ROOT="/root/autodl-tmp/"
DATR_DIR=$ROOT/Data/chatgpt/processed
MAIN=$ROOT/Code/RLHF/src/train_sft.py
MODEL_PATH=$ROOT/Data/models/$MODEL
OUTPUT_DIR=$ROOT/Data/chatgpt/output/sft/$MODEL
TRAIN_FILENAME="train_data_external_v1.jsonl"
EVAL_FILENAME="dev_data_external_v1.jsonl"

cd $ROOT/Code/RLHF || exit
mkdir -p $OUTPUT_DIR

#CUDA_VISIBLE_DEVICES=1 deepspeed --master_port 5008 $MAIN \
#deepspeed --num_gpus 1 $MAIN \
python $MAIN \
  --data_dir $DATR_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_PATH \
  --max_length 512 \
  --logging_steps 100 \
  --do_train \
  --train_filename $TRAIN_FILENAME \
  --train_batch_size 8 \
  --gradient_accumulation_steps 16 \
  --save_strategy "steps" \
  --save_steps 1000 \
  --num_epochs 1 \
  --deepspeed_config "ds_config_sft_pangu.json" \
  --do_eval \
  --eval_filename $EVAL_FILENAME \
  --eval_batch_size 48 \
  > out/train_sft_${MODEL}_"`date "+%Y-%m-%d-%H:%M:%S"`".log 2>&1 &
