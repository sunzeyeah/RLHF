#!/bin/bash

for TASK in "ocnli" "cmnli" "chid" "cmrc2018" "cluewsc2020" "c3" "afqmc" "csl" "iflytek" "tnews"
do
  for MODEL in "pangu-350M" "pangu-2.6B"
  do
    ROOT="/mnt/private-pa002-vol726121-prd/"
    #ROOT="/root/autodl-tmp/"
    DATR_DIR=$ROOT/Data/chatgpt/raw/$TASK
    MAIN=$ROOT/Code/RLHF/src/eval_pretrain.py
    MODEL_PATH=$ROOT/Data/models/$MODEL
    OUTPUT_DIR=$ROOT/Data/chatgpt/output/pretrain/$MODEL
    EVAL_FILENAME="dev.json"
    if [ $MODEL == "pangu-350M" ]
    then
      BATCH_SIZE=32
    else
      BATCH_SIZE=8
    fi

    cd $ROOT/Code/RLHF || exit
    mkdir -p $OUTPUT_DIR

    #CUDA_VISIBLE_DEVICES=1 deepspeed --master_port 5008 $MAIN \
    #deepspeed --num_gpus 1 $MAIN \
    python $MAIN \
      --local_rank 0 \
      --data_dir $DATR_DIR \
      --output_dir $OUTPUT_DIR \
      --model_name_or_path $MODEL_PATH \
      --task $TASK \
      --max_length 512 \
      --do_eval \
      --eval_filename $EVAL_FILENAME \
      --eval_batch_size $BATCH_SIZE \
      > eval_pretrain_${MODEL}_${TASK}_"`date "+%Y-%m-%d-%H:%M:%S"`".log 2>&1
  done
done