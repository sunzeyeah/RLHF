#!/bin/bash

MODEL="pangu-2.6B"

#ROOT="/mnt/sfevol775196/sunzeye273/sunzeye273"
#ROOT="/mnt/pa002-28359-vol543625-private"
ROOT="/root/autodl-tmp"
DATR_DIR=$ROOT/Data/chatgpt/processed
MAIN=$ROOT/Code/RLHF/src/train_sft.py
MODEL_PATH=$ROOT/Data/models/$MODEL
#MODEL_PATH=/mnt/pa002-28359-vol543625-share/LLM-data/checkpoint/$MODEL
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
  --do_train \
  --train_filename $TRAIN_FILENAME \
  --train_batch_size 8 \
  --gradient_accumulation_steps 16 \
  --num_epochs 1 \
  --lora_rank 100 \
  --do_eval \
  --eval_filename $EVAL_FILENAME \
  --eval_batch_size 48 \
  > out/train_sft_${MODEL}_"`date "+%Y-%m-%d-%H:%M:%S"`".log 2>&1 &
