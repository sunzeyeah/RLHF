#!/bin/bash

#ROOT="/Users/zeyesun/Documents/"
ROOT="/root/autodl-tmp/"
DATR_DIR=$ROOT/Data/rlhf/raw
MAIN=$ROOT/Code/RLHF/src/train_sft.py
MODEL_PATH=$ROOT/Data/models/pangu-350M
OUTPUT_DIR=$ROOT/Data/rlhf/output/sft
TRAIN_FILENAME="baike_qa_train.json"
EVAL_FILENAME="baike_qa_valid.json"

cd $ROOT/Code/RLHF || exit

#python $MAIN \
#CUDA_VISIBLE_DEVICES=1 deepspeed --master_port 5008 $MAIN \
deepspeed --num_gpus 1 $MAIN \
  --data_dir $DATR_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_PATH \
  --do_train \
  --train_filename $TRAIN_FILENAME \
  --max_length 512 \
  --train_batch_size 4 \
  --num_epochs 1 \
  --logging_steps 10 \
  --do_eval \
  --eval_filename $EVAL_FILENAME \
  > train_sft.log 2>&1 &
