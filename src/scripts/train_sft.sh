#!/bin/bash

ROOT="/Users/zeyesun/Documents/"
DATR_DIR=$ROOT/Data/rlhf/raw
MAIN=$ROOT/Code/RLHF/src/train_sft.py
MODEL_PATH=$ROOT/Data/models/pangu-350M
OUTPUT_DIR=$ROOT/Data/rlhf/output/sft
TRAIN_FILENAME="baike_qa_train_small.json"
EVAL_FILENAME="baike_qa_valid_small.json"

cd $ROOT/Code/RLHF || exit

#CUDA_VISIBLE_DEVICES=1,2 deepspeed --master_port 5008 $MAIN \
python $MAIN \
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
