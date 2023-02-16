#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 deepspeed --master_port 5009 train_gptj_summarize.py > train_CPM_dialogue_20230216.log 2>&1 &
tail -f train_CPM_dialogue_20230216.log