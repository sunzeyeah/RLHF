#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2 deepspeed --master_port 5008 train_reward_model.py > train_pangu_rm.log 2>&1 &
