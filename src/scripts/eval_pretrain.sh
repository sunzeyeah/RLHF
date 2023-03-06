#!/bin/bash

MODEL="pangu-2.6B"
TASK="ocnli"

ROOT="/mnt/private-pa002-vol726121-prd/"
#ROOT="/root/autodl-tmp/"
DATR_DIR=$ROOT/Data/chatgpt/raw/$TASK
MAIN=$ROOT/Code/RLHF/src/eval_pretrain.py
MODEL_PATH=$ROOT/Data/models/$MODEL
OUTPUT_DIR=$ROOT/Data/chatgpt/output/pretrain/$MODEL
#TRAIN_FILENAME="train.json"
EVAL_FILENAME="dev.json"

cd $ROOT/Code/RLHF || exit
mkdir -p $OUTPUT_DIR

#CUDA_VISIBLE_DEVICES=1 deepspeed --master_port 5008 $MAIN \
#deepspeed --num_gpus 1 $MAIN \
python $MAIN \
  --data_dir $DATR_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_PATH \
  --task $TASK \
  --max_length 512 \
  --deepspeed_config "ds_config_pretrain_pangu.json" \
  --do_eval \
  --eval_filename $EVAL_FILENAME \
  --eval_batch_size 8 \
  > eval_pretrain_${MODEL}_${TASK}_"`date "+%Y-%m-%d-%H:%M:%S"`".log 2>&1 &
