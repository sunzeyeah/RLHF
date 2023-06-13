#!/bin/bash

MODEL="llama-7B"

ROOT="/mnt/sfevol775196/sunzeye273"
#ROOT="/mnt/share-pa002-vol682688-prd/sunzeye273"
#ROOT="/mnt/pa002-28359-vol543625-private"
#ROOT="/root/autodl-tmp"
DATR_DIR=$ROOT/Data/chatgpt/processed
#MAIN=$ROOT/Code/chatgpt/src/pretrain.py
MAIN=$ROOT/Code/RLHF/src/pretrain_wo_trainer.py
MODEL_PATH=$ROOT/Data/models/$MODEL
#MODEL_PATH=/mnt/pa002-28359-vol543625-share/LLM-data/checkpoint/$MODEL
OUTPUT_DIR=$ROOT/Data/chatgpt/output/pretrain/$MODEL
TRAIN_FILENAME="train_data_v1.jsonl"

#cd $ROOT/Code/chatgpt || exit
cd $ROOT/Code/RLHF || exit
mkdir -p $OUTPUT_DIR

#CUDA_VISIBLE_DEVICES=1 deepspeed --master_port 5008 $MAIN \
#python $MAIN \
CUDA_LAUNCH_BLOCKING=1 deepspeed $MAIN \
  --data_dir $DATR_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_PATH \
  --max_length 2048 \
  --logging_steps 10 \
  --save_steps 50 \
  --learning_rate 1e-5 \
  --do_train \
  --train_filename $TRAIN_FILENAME \
  --num_epochs 2 \
  --train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type "WarmupLR" \
  --gradient_checkpointing \
  --deepspeed_config "stage-3-no_trainer.json" \
  > out/pretrain_${MODEL}_"`date "+%Y-%m-%d-%H:%M:%S"`".log 2>&1 &
