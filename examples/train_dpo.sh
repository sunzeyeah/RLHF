#!/bin/bash

MODEL="chatglm2-6B"

#ROOT="/mnt/sfevol775196/sunzeye273"
ROOT="/mnt/pa002-28359-vol543625-private"
#ROOT="/root/autodl-tmp"
DATR_DIR=$ROOT/Data/chatgpt/processed
#MAIN=$ROOT/Code/chatgpt/src/train_dpo.py
MAIN=$ROOT/Code/RLHF/src/train_dpo.py
#TOKENIZER_PATH=$ROOT/Data/models/$MODEL
TOKENIZER_PATH=/mnt/pa002-28359-vol543625-share/LLM-data/checkpoint/$MODEL
MODEL_PATH=$ROOT/Data/chatgpt/output/sft/$MODEL
REFERENCE_MODEL_PATH=$ROOT/Data/chatgpt/output/sft/$MODEL
OUTPUT_DIR=$ROOT/Data/chatgpt/output/dpo/$MODEL
TRAIN_FILENAME="sft_train_v2.1.jsonl"
EVAL_FILENAME="sft_eval_v2.1.jsonl"
TEST_FILENAME="sft_star_v2.1.jsonl"
OUTPUT_FILENAME="dpo_logps_v2.1.bin"

#cd $ROOT/Code/chatgpt || exit
cd $ROOT/Code/RLHF || exit
mkdir -p $OUTPUT_DIR

if [ -f $OUTPUT_DIR/$OUTPUT_FILENAME ]
then
    echo "${OUTPUT_DIR}/${OUTPUT_FILENAME} already exists, skipping prediction stage"
else
    python $MAIN \
      --local_rank 0 \
      --device_map "cuda:0" \
      --data_dir $DATR_DIR \
      --output_dir $OUTPUT_DIR \
      --tokenizer_path $TOKENIZER_PATH \
      --model_name_or_path $MODEL_PATH \
      --max_length 512 \
      --logging_steps 10 \
      --eval_batch_size 32 \
      --do_pred \
      --test_filename $TEST_FILENAME \
      --output_filename $OUTPUT_FILENAME \
      > out/pred_dpo_${MODEL}_"`date "+%Y-%m-%d-%H:%M:%S"`".log
fi

#CUDA_VISIBLE_DEVICES=1 deepspeed --master_port 5008 $MAIN \
#python $MAIN \
CUDA_LAUNCH_BLOCKING=1 deepspeed $MAIN \
  --data_dir $DATR_DIR \
  --output_dir $OUTPUT_DIR \
  --tokenizer_path $TOKENIZER_PATH \
  --model_name_or_path $MODEL_PATH \
  --max_length 512 \
  --logging_steps 10 \
  --save_steps 100 \
  --learning_rate 1e-5 \
  --do_train \
  --train_filename $TRAIN_FILENAME \
  --train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --num_epochs 5 \
  --gradient_checkpointing \
  --deepspeed_config "stage-3.json" \
  --do_eval \
  --eval_filename $EVAL_FILENAME \
  --eval_batch_size 32 \
  --output_filename $OUTPUT_FILENAME \
  > out/train_dpo_${MODEL}_"`date "+%Y-%m-%d-%H:%M:%S"`".log 2>&1 &
