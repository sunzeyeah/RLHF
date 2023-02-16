#!/bin/bash

Time=$(date "+%Y%m%d-%H%M%S")
log_file=log_dir/debug_ppo_RLHF_${Time}.log
accelerate launch --main_process_port 5007 --config_file configs/default_accelerate_config.yaml trlx_gptj_text_summarization.py > $log_file 2>&1 &
echo $log_file