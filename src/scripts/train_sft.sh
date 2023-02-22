#!/bin/bash

MODEL="pangu-350M"

#ROOT="/Users/zeyesun/Documents/"
ROOT="/root/autodl-tmp/"
DATR_DIR=$ROOT/Data/raw
MAIN=$ROOT/Code/RLHF/src/train_sft.py
MODEL_PATH=$ROOT/Data/models/$MODEL
OUTPUT_DIR=$ROOT/Data/output/sft/$MODEL
TRAIN_FILENAME="baike_qa_train.json"
EVAL_FILENAME="baike_qa_valid.json"

cd $ROOT/Code/RLHF || exit
mkdir -p $OUTPUT_DIR

#python $MAIN \
#CUDA_VISIBLE_DEVICES=1 deepspeed --master_port 5008 $MAIN \
deepspeed --num_gpus 1 $MAIN \
  --data_dir $DATR_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_PATH \
  --max_length 1024 \
  --logging_steps 100 \
  --do_train \
  --train_filename $TRAIN_FILENAME \
  --train_batch_size 24 \
  --gradient_accumulation_steps 4 \
  --save_strategy "steps" \
  --save_steps 5000 \
  --num_epochs 1 \
  --deepspeed_config "ds_config_sft_pangu.json" \
  --do_eval \
  --eval_filename $EVAL_FILENAME \
  --eval_batch_size 48 \
  > train_sft_${MODEL}_"`date "+%Y-%m-%d-%H:%M:%S"`".log 2>&1 &
