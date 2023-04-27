# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import torch
import deepspeed
import math
import json

from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, get_scheduler
from transformers import AutoConfig, AutoModel
from transformers.deepspeed import HfDeepSpeedConfig

from src.utils.config import get_train_ds_config, get_eval_ds_config
# from utils.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters
from src.models import RewardModel
from src.utils.logger import logger, RESOURCE_PATH
from src.utils.modeling_utils import get_optimizer_grouped_parameters
"""
TODOs:
  * support HF models for critic (for debugging), must be a previously saved ckpt from step-2
  * determine ds_config/zero_stage based on model size, gpu style, world size, etc
    - get model size by creating simple meta model
    - 1.3b: zero-2 for actor/ref models, zero-0 for others
    - 13b+: zero-3 for all models
"""


def log_init(model_name, rank, stime=None):
    if rank == 0:
        tag = "start" if stime is None else "end"
        suffix = "ing" if stime is None else "ed"
        duration = ""
        if stime is not None:
            duration = "(duration: {:.2f}s)".format(time.time() - stime)
        logger.info(f"[{tag}] Initializ{suffix} {model_name} Model [{tag}] {duration}")

        return time.time()


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    disable_dropout=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config, trust_remote_code=True)
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config, trust_remote_code=True)

    model.config.end_token_id = tokenizer.eos_token_id
    # model.config.pad_token_id = model.config.eos_token_id
    # model.resize_token_embeddings(int(
    #     8 *
    #     math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model


def create_critic_model(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        disable_dropout=False,
                        checkpoint=None,
                        lora_rank=0,
                        lora_alpha=1,
                        lora_train_bias="none"):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule
    if "pangu" in model_name_or_path:
        model_class = AutoModelForCausalLM
    elif "glm" in model_name_or_path:
        model_class = AutoModelForSeq2SeqLM
    else:
        raise ValueError(f"Unsupported model type: {model_name_or_path}")
    critic_model = create_hf_model(model_class, model_name_or_path, tokenizer,
                                   ds_config, rlhf_training, disable_dropout)
    critic_model.config.lora_rank = lora_rank
    critic_model.config.lora_alpha = lora_alpha
    critic_model.config.lora_train_bias = lora_train_bias
    critic_model = RewardModel(critic_model.config, critic_model.transformer, tokenizer,
        # num_padding_at_beginning=num_padding_at_beginning
     )

    if rlhf_training:
        assert os.path.exists(checkpoint), f"Cannot find reward model checkpoint at {checkpoint}"
        critic_model.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    return critic_model


class DeepSpeedRLHFEngine:

    def __init__(self, actor_model_name_or_path, critic_model_name_or_path,
                 tokenizer, args, num_total_iters):
        self.args = args
        self.num_total_iters = num_total_iters
        self.tokenizer = tokenizer
        if "pangu" in actor_model_name_or_path:
            self.model_class = AutoModelForCausalLM
        elif "glm" in actor_model_name_or_path:
            self.model_class = AutoModelForSeq2SeqLM
        else:
            raise ValueError(f"Unsuppported model type: {actor_model_name_or_path}")

        self.actor = self._init_actor(
            actor_model_name_or_path=actor_model_name_or_path)
        self.ref = self._init_ref(
            actor_model_name_or_path=actor_model_name_or_path)
        self.actor_ema = None
        if self.args.enable_ema:
            self.actor_ema = self._init_ema(
                actor_model_name_or_path=actor_model_name_or_path)

        self.critic = self._init_critic(
            critic_model_name_or_path=critic_model_name_or_path)
        self.reward = self._init_reward(
            critic_model_name_or_path=critic_model_name_or_path)
        if self.args.critic_gradient_checkpointing:
            self.critic.gradient_checkpointing_enable()

    def _init_actor(self, actor_model_name_or_path):
        stime = log_init("Actor", self.args.local_rank)

        # DS Config
        ds_config = get_train_ds_config(
            global_batch_size=self.args.global_train_batch_size_actor,
            micro_batch_size=self.args.ppo_train_batch_size,
            offload=self.args.offload,
            stage=self.args.actor_zero_stage,
            enable_hybrid_engine=self.args.enable_hybrid_engine,
            inference_tp_size=self.args.inference_tp_size,
            release_inference_cache=self.args.release_inference_cache,
            pin_parameters=(not self.args.unpin_actor_parameters),
            tp_gather_partition_size=self.args.tp_gather_partition_size,
            max_out_tokens=self.args.max_length)

        # Model
        actor_model = create_hf_model(
            model_class=self.model_class,
            model_name_or_path=actor_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_config,
            disable_dropout=self.args.disable_actor_dropout)

        # LoRA
        if self.args.actor_lora_rank > 0:
            actor_model = convert_linear_layer_to_lora(
                actor_model, self.args.actor_lora_module_name,
                self.args.actor_lora_rank)
            if self.args.only_optimize_lora:
                actor_model = only_optimize_lora_parameters(actor_model)

        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(
            actor_model, self.args.actor_weight_decay)
        optim = AdamOptimizer(optim_params,
                              lr=self.args.actor_learning_rate,
                              betas=(0.9, 0.95))

        # LR Scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        actor_engine, *_ = deepspeed.initialize(model=actor_model,
                                                optimizer=optim,
                                                lr_scheduler=lr_scheduler,
                                                config=ds_config)
        actor_engine.config['pad_token_id'] = actor_model.config.pad_token_id
        log_init("Actor", self.args.local_rank, stime=stime)

        return actor_engine

    def _init_ref(self, actor_model_name_or_path):
        stime = log_init("Ref", self.args.local_rank)
        # DS Config
        zero_stage = self.args.actor_zero_stage
        if zero_stage != 3:
            # If actor is ZeRO-3 then we use it for everything, otherwise assume we have enough memory for ref model
            zero_stage = 0
        ds_config = get_eval_ds_config(global_batch_size=self.args.global_train_batch_size_actor,
                                       micro_batch_size=self.args.ppo_train_batch_size,
                                       offload=self.args.offload_reference_model,
                                       stage=zero_stage)

        ref_model = create_hf_model(self.model_class,
                                    actor_model_name_or_path, self.tokenizer,
                                    ds_config)

        ref_engine, *_ = deepspeed.initialize(model=ref_model,
                                              config=ds_config)

        log_init("Ref", self.args.local_rank, stime=stime)
        return ref_engine

    def _init_ema(self, actor_model_name_or_path):
        stime = log_init("EMA", self.args.local_rank)
        # DS Config
        zero_stage = self.args.actor_zero_stage
        if zero_stage != 3:
            # If actor is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            zero_stage = 0
        ds_config = get_eval_ds_config(global_batch_size=self.args.global_train_batch_size_actor,
                                       micro_batch_size=self.args.ppo_train_batch_size,
                                       offload=self.args.offload_reference_model,
                                       stage=zero_stage)

        actor_model_ema = create_hf_model(self.model_class,
                                          actor_model_name_or_path,
                                          self.tokenizer, ds_config)
        if self.args.actor_lora_rank > 0:
            actor_model_ema = convert_linear_layer_to_lora(
                actor_model_ema, self.args.actor_lora_module_name,
                self.args.actor_lora_rank)

        ema_engine, *_ = deepspeed.initialize(model=actor_model_ema,
                                              config=ds_config)

        log_init("EMA", self.args.local_rank, stime=stime)
        return ema_engine

    def _init_critic(self, critic_model_name_or_path):
        stime = log_init("Critic", self.args.local_rank)
        ds_config = get_train_ds_config(global_batch_size=self.args.global_train_batch_size_critic,
                                        micro_batch_size=self.args.ppo_train_batch_size,
                                        offload=self.args.offload,
                                        stage=self.args.critic_zero_stage)

        #TODO(jeff): should not be needed, we should be able to use ds_config above
        #TODO(jeff): it means we never create the critic w. zero.init context if we are using ZeRO-3
        ds_eval_config = get_eval_ds_config(global_batch_size=self.args.global_train_batch_size_critic,
                                            micro_batch_size=self.args.ppo_train_batch_size,
                                            offload=False,
                                            stage=0)

        # Model
        critic_model = create_critic_model(
            model_name_or_path=critic_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_eval_config,
            # num_padding_at_beginning=self.args.num_padding_at_beginning,
            rlhf_training=True,
            disable_dropout=self.args.disable_critic_dropout,
            checkpoint=self.args.critic_checkpoint,
            lora_rank=self.args.critic_lora_rank,
            lora_alpha=self.args.lora_alpha,
            lora_train_bias=self.args.lora_train_bias)

        # LoRA
        if self.args.critic_lora_rank > 0:
            critic_model = convert_linear_layer_to_lora(
                critic_model, self.args.critic_lora_module_name,
                self.args.critic_lora_rank)
            if self.args.only_optimize_lora:
                critic_model = only_optimize_lora_parameters(critic_model)

        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        optim_pararms = get_optimizer_grouped_parameters(
            critic_model, self.args.critic_weight_decay)
        optim = AdamOptimizer(optim_pararms,
                              lr=self.args.critic_learning_rate,
                              betas=(0.9, 0.95))

        # LR Scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        critic_engine, *_ = deepspeed.initialize(model=critic_model,
                                                 optimizer=optim,
                                                 lr_scheduler=lr_scheduler,
                                                 config=ds_config)

        log_init("Critic", self.args.local_rank, stime=stime)
        return critic_engine

    def _init_reward(self, critic_model_name_or_path):
        stime = log_init("Reward", self.args.local_rank)
        # DS Config
        zero_stage = self.args.critic_zero_stage
        if zero_stage != 3:
            # If critic is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            zero_stage = 0
        ds_config = get_eval_ds_config(global_batch_size=self.args.global_train_batch_size_critic,
                                       micro_batch_size=self.args.ppo_train_batch_size,
                                       offload=self.args.offload,
                                       stage=zero_stage)

        #TODO(jeff): should not be needed, we should be able to use ds_config above
        #TODO(jeff): it means we never create the critic w. zero.init context if we are using ZeRO-3
        ds_eval_config = get_eval_ds_config(global_batch_size=self.args.global_train_batch_size_critic,
                                            micro_batch_size=self.args.ppo_train_batch_size,
                                            offload=False,
                                            stage=0)

        # Model
        reward_model = create_critic_model(
            model_name_or_path=critic_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_eval_config,
            # num_padding_at_beginning=self.args.num_padding_at_beginning,
            rlhf_training=True,
            checkpoint=self.args.critic_checkpoint,
            lora_rank=self.args.critic_lora_rank,
            lora_alpha=self.args.lora_alpha,
            lora_train_bias=self.args.lora_train_bias)

        reward_engine, *_ = deepspeed.initialize(model=reward_model,
                                                 config=ds_config)

        log_init("Reward", self.args.local_rank, stime=stime)
        return reward_engine
