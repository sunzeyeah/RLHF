#!/bin/bash

REWARD_MODEL="pangu-350M"
SFT_MODEL="pangu-2.6B"

ROOT="/mnt/sfevol775196/sunzeye273"
#ROOT="/mnt/share-pa002-vol682688-prd/sunzeye273"
#ROOT="/mnt/pa002-28359-vol543625-private"
#ROOT="/root/autodl-tmp/"
DATR_DIR=$ROOT/Data/chatgpt/processed
#MAIN=$ROOT/Code/chatgpt/src/train_rlhf.py
MAIN=$ROOT/Code/RLHF/src/train_rlhf.py
#ACCELERATE_CONFIG=$ROOT/Code/RLHF/src/resources/ppo_model/default_accelerate_config.yaml
TOKENIZER_PATH=$ROOT/Data/models/$REWARD_MODEL
REWARD_MODEL_PATH=$ROOT/Data/models/$REWARD_MODEL
#REWARD_MODEL_PATH=/mnt/pa002-28359-vol543625-share/LLM-data/checkpoint/$REWARD_MODEL
SFT_MODEL_PATH=$ROOT/Data/chatgpt/output/sft/$SFT_MODEL
REWARD_CHECKPOINT=$ROOT/Data/chatgpt/output/reward/$REWARD_MODEL/pytorch_model.bin
OUTPUT_DIR=$ROOT/Data/chatgpt/output/rlhf/$SFT_MODEL
TRAIN_FILENAME="train_data_external_v1.jsonl"
EVAL_FILENAME="dev_data_external_v1.jsonl"

#cd $ROOT/Code/chatgpt || exit
cd $ROOT/Code/RLHF || exit
mkdir -p $OUTPUT_DIR

#python $MAIN \
#accelerate launch --main_process_port 5007 --config_file $ACCELERATE_CONFIG $MAIN \
python $MAIN \
  --data_dir $DATR_DIR \
  --output_dir $OUTPUT_DIR \
  --tokenizer_path $TOKENIZER_PATH \
  --sft_model_path $SFT_MODEL_PATH \
  --reward_model_path $REWARD_MODEL_PATH \
  --reward_checkpoint $REWARD_CHECKPOINT \
  --max_length 512 \
  --logging_steps 10 \
  --do_train \
  --train_filename $TRAIN_FILENAME \
  --train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --num_epochs 1 \
  --ppo_config "ppo_config_${SFT_MODEL}.yml" \
  --do_eval \
  --eval_filename $EVAL_FILENAME \
  --eval_batch_size 16 \
  > out/train_rlhf_${SFT_MODEL}_"`date "+%Y-%m-%d-%H:%M:%S"`".log 2>&1 &
