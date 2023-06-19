#!/bin/bash

MODEL="llama-7B"

ROOT="/mnt/sfevol775196/sunzeye273"
#ROOT="/mnt/share-pa002-vol682688-prd/sunzeye273"
#ROOT="/mnt/pa002-28359-vol543625-private"
#ROOT="/root/autodl-tmp"
DATR_DIR=$ROOT/Data/chatgpt/processed
#MAIN=$ROOT/Code/chatgpt/src/pretrain.py
MAIN=$ROOT/Code/RLHF/src/pretrain.py
MODEL_PATH=$ROOT/Data/models/$MODEL
#MODEL_PATH=/mnt/pa002-28359-vol543625-share/LLM-data/checkpoint/$MODEL
OUTPUT_DIR=$ROOT/Data/chatgpt/output/pretrain/$MODEL/checkpoint-2000
CHECKPOINT=$OUTPUT_DIR/pytorch_model.bin
TEST_FILENAME="test_prompts.jsonl"
OUTPUT_FILENAME="output_${MODEL}.jsonl"

#cd $ROOT/Code/chatgpt || exit
cd $ROOT/Code/RLHF || exit
mkdir -p $OUTPUT_DIR

#CUDA_VISIBLE_DEVICES=1 deepspeed --master_port 5008 $MAIN \
python $MAIN \
  --local_rank 0 \
  --data_dir $DATR_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_PATH \
  --checkpoint $CHECKPOINT \
  --max_length 1024 \
  --max_length_generation 512 \
  --bits 16 \
  --do_pred \
  --test_filename $TEST_FILENAME \
  --output_filename $OUTPUT_FILENAME \
  > out/pred_pretrain_${MODEL}_"`date "+%Y-%m-%d-%H:%M:%S"`".log 2>&1 &
