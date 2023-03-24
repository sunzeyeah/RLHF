#!/bin/bash

for TASK in  "cluewsc2020" "afqmc" "csl" "iflytek" "ocnli" "cmnli" "tnews" "c3" "cmrc2018" "chid"
do
  for MODEL in "pangu-350M" "pangu-2.6B" "glm-350M-chinese" "glm-10B-chinese" "pangu-13B"
  do
    ROOT="/mnt/sfevol775196"
    #ROOT="/mnt/pa002-28359-vol543625-private"
    #ROOT="/root/autodl-tmp"
    DATR_DIR=$ROOT/Data/chatgpt/raw/$TASK
    MAIN=$ROOT/Code/RLHF/src/eval_pretrain.py
    MODEL_PATH=$ROOT/Data/models/$MODEL
    OUTPUT_DIR=$ROOT/Data/chatgpt/output/pretrain/$MODEL
    EVAL_FILENAME="dev.json"
    TRAIN_FILENAME="train.json"
    case $MODEL in
       "pangu-2.6B")
          BATCH_SIZE=8
          ;;
       *)
         BATCH_SIZE=32
         ;;
    esac

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
      --train_filename $TRAIN_FILENAME \
      --eval_filename $EVAL_FILENAME \
      --eval_batch_size $BATCH_SIZE \
      --top_p 0.8 \
      --temperature 0.8 \
      --num_return_sequences 1 \
      --max_length_generation 100 \
      > out/eval_pretrain_${MODEL}_${TASK}_"`date "+%Y-%m-%d-%H:%M:%S"`".log 2>&1
  done
done