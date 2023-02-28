#!/bin/bash

MODEL="pangu-350M"

#ROOT="/mnt/private-pa002-vol726121-prd/"
ROOT="/root/autodl-tmp/"
DATR_DIR=$ROOT/Data/raw
MAIN=$ROOT/Code/RLHF/src/train_sft.py
MODEL_PATH=$ROOT/Data/models/$MODEL
OUTPUT_DIR=$ROOT/Data/output/sft/$MODEL
TEST_FILENAME=""
OUTPUT_FILENAME=""
CHECKPOINT=$OUTPUT_DIR

cd $ROOT/Code/RLHF || exit
mkdir -p $OUTPUT_DIR

#CUDA_VISIBLE_DEVICES=1 deepspeed --master_port 5008 $MAIN \
#deepspeed --num_gpus 1 $MAIN \
python $MAIN \
  --data_dir $DATR_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_PATH \
  --checkpoint $CHECKPOINT \
  --max_length 1024 \
  --logging_steps 10 \
  --do_pred \
  --test_filename $TEST_FILENAME \
  --output_filename $OUTPUT_FILENAME \
  --eval_batch_size 96 \
  > pred_sft_${MODEL}.log 2>&1 &
