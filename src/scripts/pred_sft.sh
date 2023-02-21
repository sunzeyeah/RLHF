#!/bin/bash

MODEL="pangu-2.6B"

#ROOT="/Users/zeyesun/Documents/"
ROOT="/root/autodl-tmp/"
DATR_DIR=$ROOT/Data/rlhf/raw
MAIN=$ROOT/Code/RLHF/src/train_sft.py
MODEL_PATH=$ROOT/Data/models/$MODEL
OUTPUT_DIR=$ROOT/Data/rlhf/output/sft/$MODEL
TEST_FILENAME=""
OUTPUT_FILENAME=""
CHECKPOINT=$OUTPUT_DIR

cd $ROOT/Code/RLHF || exit
mkdir -p $OUTPUT_DIR

#python $MAIN \
#CUDA_VISIBLE_DEVICES=1 deepspeed --master_port 5008 $MAIN \
deepspeed --num_gpus 1 $MAIN \
  --data_dir $DATR_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_PATH \
  --checkpoint $CHECKPOINT \
  --max_length 512 \
  --do_pred \
  --test_filename $TEST_FILENAME \
  --output_filename $OUTPUT_FILENAME \
  --logging_steps 10 \
  --eval_batch_size 16 \
  > pred_sft_${MODEL}.log 2>&1 &
