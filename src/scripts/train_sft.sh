#!/bin/bash

MODEL="pangu-2.6B"

#ROOT="/Users/zeyesun/Documents/"
ROOT="/root/autodl-tmp/"
DATR_DIR=$ROOT/Data/rlhf/raw
MAIN=$ROOT/Code/RLHF/src/train_sft.py
MODEL_PATH=$ROOT/Data/models/$MODEL
OUTPUT_DIR=$ROOT/Data/rlhf/output/sft/$MODEL
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
  --max_length 512 \
  --logging_steps 100 \
  --do_train \
  --train_filename $TRAIN_FILENAME \
  --train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_epochs 1 \
  --do_eval \
  --eval_filename $EVAL_FILENAME \
  --eval_batch_size 16 \
  > train_sft_${MODEL}_"`date "+Y%-%m-%d-%H:%M:%S"`".log 2>&1 &
