#!/bin/bash

MODEL="pangu-350M"

ROOT="/mnt/sfevol775196/sunzeye273"
#ROOT="/mnt/pa002-28359-vol543625-private"
#ROOT="/root/autodl-tmp"
DATR_DIR=$ROOT/Data/chatgpt/processed
MAIN=$ROOT/Code/RLHF/src/train_reward.py
TOKENIZER_PATH=$ROOT/Data/models/$MODEL
#TOKENIZER_PATH=/mnt/pa002-28359-vol543625-share/LLM-data/checkpoint/$MODEL
MODEL_PATH=$ROOT/Data/chatgpt/output/sft/${MODEL}
OUTPUT_DIR=$ROOT/Data/chatgpt/output/reward/$MODEL
TEST_FILENAME="test_data_external_v1.jsonl"
OUTPUT_FILENAME="result_external_v1.tsv"
CHECKPOINT="${ROOT}/Data/chatgpt/output/reward/${MODEL}/pytorch_modelstar.bin"

cd $ROOT/Code/RLHF || exit
mkdir -p $OUTPUT_DIR

#python $MAIN \
CUDA_LAUNCH_BLOCKING=1 deepspeed --num_gpus 1 $MAIN \
  --data_dir $DATR_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_PATH \
  --tokenizer_path $TOKENIZER_PATH \
  --checkpoint $CHECKPOINT \
  --max_length 512 \
  --logging_steps 100 \
  --deepspeed_config "reward_stage-2_${MODEL}_v100.json" \
  --do_pred \
  --test_filename $TEST_FILENAME \
  --eval_batch_size 96 \
  --output_filename $OUTPUT_FILENAME \
  > out/pred_reward_${MODEL}_"`date "+%Y-%m-%d-%H:%M:%S"`".log 2>&1 &
