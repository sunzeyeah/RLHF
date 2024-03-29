#!/bin/bash

# C-Eval and MMLU benchamarks
TASK="ceval"
MODEL="llama-7B"
ROOT="/mnt/sfevol775196/sunzeye273"
#ROOT="/mnt/share-pa002-vol682688-prd/sunzeye273"
#ROOT="/mnt/pa002-28359-vol543625-private"
#ROOT="/root/autodl-tmp"
DATR_DIR=$ROOT/Data/chatgpt/raw/$TASK
#MAIN=$ROOT/Code/chatgpt/src/eval_pretrain.py
MAIN=$ROOT/Code/RLHF/src/eval_pretrain.py
MODEL_PATH=$ROOT/Data/models/$MODEL
#MODEL_PATH=/mnt/pa002-28359-vol543625-share/LLM-data/checkpoint/$MODEL
OUTPUT_DIR=$ROOT/Data/chatgpt/output/pretrain/$MODEL
EVAL_FILENAME="val"
TRAIN_FILENAME="dev"
CHECKPOINT=$ROOT/Data/chatgpt/output/pretrain/$MODEL
SHOTS=5
MAX_LENGTH=1280

cd $ROOT/Code/RLHF || exit
#    cd $ROOT/Code/chatgpt || exit
mkdir -p $OUTPUT_DIR

#CUDA_VISIBLE_DEVICES=1 deepspeed --master_port 5008 $MAIN \
#deepspeed --num_gpus 1 $MAIN \
python $MAIN \
  --device_map "auto" \
  --data_dir $DATR_DIR \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_PATH \
  --task $TASK \
  --train_filename $TRAIN_FILENAME \
  --eval_filename $EVAL_FILENAME \
  --checkpoint $CHECKPOINT \
  --max_length $MAX_LENGTH \
  --max_few_shot $SHOTS \
  --max_length_generation 1 \
  > out/eval_pretrain_${TASK}_${MODEL}_${EVAL_FILENAME}_${SHOTS}-shots_${MAX_LENGTH}_"`date "+%Y-%m-%d-%H:%M:%S"`".log 2>&1 &

## Traditional NLP benchmark Evaluations
#for TASK in  "cluewsc2020" "afqmc" "csl" "iflytek" "ocnli" "cmnli" "tnews" "c3" "cmrc2018" "chid"
#do
#  for MODEL in "pangu-350M" "pangu-2.6B" "glm-350M-chinese" "glm-10B-chinese" "pangu-13B"
#  do
#    ROOT="/mnt/sfevol775196/sunzeye273"
#    #ROOT="/mnt/share-pa002-vol682688-prd/sunzeye273"
#    #ROOT="/mnt/pa002-28359-vol543625-private"
#    #ROOT="/root/autodl-tmp"
#    DATR_DIR=$ROOT/Data/chatgpt/raw/$TASK
##    MAIN=$ROOT/Code/chatgpt/src/eval_pretrain.py
#    MAIN=$ROOT/Code/RLHF/src/eval_pretrain.py
#    MODEL_PATH=$ROOT/Data/models/$MODEL
#    #MODEL_PATH=/mnt/pa002-28359-vol543625-share/LLM-data/checkpoint/$MODEL
#    OUTPUT_DIR=$ROOT/Data/chatgpt/output/pretrain/$MODEL
#    EVAL_FILENAME="dev.json"
#    TRAIN_FILENAME="train.json"
#    case $MODEL in
#       "pangu-2.6B")
#          BATCH_SIZE=8
#          ;;
#       *)
#         BATCH_SIZE=32
#         ;;
#    esac
#
#    cd $ROOT/Code/RLHF || exit
##    cd $ROOT/Code/chatgpt || exit
#    mkdir -p $OUTPUT_DIR
#
#    #CUDA_VISIBLE_DEVICES=1 deepspeed --master_port 5008 $MAIN \
#    #deepspeed --num_gpus 1 $MAIN \
#    python $MAIN \
#      --device_map auto \
#      --data_dir $DATR_DIR \
#      --output_dir $OUTPUT_DIR \
#      --model_name_or_path $MODEL_PATH \
#      --task $TASK \
#      --max_length 512 \
#      --train_filename $TRAIN_FILENAME \
#      --eval_filename $EVAL_FILENAME \
#      --eval_batch_size $BATCH_SIZE \
#      --top_p 0.8 \
#      --temperature 0.8 \
#      --num_return_sequences 1 \
#      --max_length_generation 100 \
#      > out/eval_pretrain_${MODEL}_${TASK}_"`date "+%Y-%m-%d-%H:%M:%S"`".log 2>&1
#  done
#done