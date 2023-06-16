
import sys
sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")
sys.path.insert(0, "/mnt/sfevol775196/sunzeye273/Code/chatgpt")
# sys.path.insert(0, "/mnt/share-pa002-vol682688-prd/sunzeye273/Code/chatgpt")
sys.path.insert(0, "/mnt/pa002-28359-vol543625-private/Code/chatgpt")

import os
import argparse
import torch
import random
import copy
import deepspeed
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, default_data_collator
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader

from src.utils import logger, RESOURCE_PATH
from src.models.rlhf_engine import DeepSpeedRLHFEngine
from src.models.trainer import DeepSpeedPPOTrainer, DeepSpeedPPOPTXTrainer
from src.utils.file_utils import set_seed, print_gpu_utilization_torch, print_gpu_utilization
from src.data.data import SFTDataset, RLHFDataset, PPODataset
from src.utils.modeling_utils import get_all_reduce_mean, save_hf_format, moving_average, save_zero_three_model


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--actor_model_path", type=str, required=True)
    parser.add_argument("--critic_model_path", type=str, required=True)
    parser.add_argument("--critic_checkpoint", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=512,
                        help="total max sequence length = max prompt length + mas generation/answer length")
    parser.add_argument("--max_gen_length", type=int, default=256,
                        help="max generation/answer length")
    # train
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--train_filename", type=str, default=None)
    parser.add_argument("--pretrain_filename", type=str, default=None,
                        help="pretraining dataset (for PPO-ptx)")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--ppo_epochs", type=int, default=1,
                        help="Number of epochs to perform ppo training for each experience")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        help="transformers.trainer_utils.SchedulerType, including:"
                             "linear, cosine, cosine_with_restarts, polynomial, constant,"
                             "constant_with_warmup")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--ppo_train_batch_size", type=int, default=4,
                        help="PPO training mini batch size (per device)")
    parser.add_argument("--ppo_batch_numbers", type=int, default=1,
                        help="number of batches for PPO training")
    parser.add_argument("--actor_weight_decay", type=float, default=0.1)
    parser.add_argument("--critic_weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=int, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument('--disable_actor_dropout', action='store_true',
                        help='Disable the dropout of the actor model.')
    parser.add_argument('--disable_critic_dropout', action='store_true',
                        help='Disable the dropout of the critic model.')
    parser.add_argument("--pretrain_coef", type=float, default=10.0,
                        help="coefficient of pretraining loss in ppo-ptx objective function")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    # deepspeed
    parser.add_argument('--enable_hybrid_engine', action='store_true',
                        help="Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed.")
    parser.add_argument('--actor_zero_stage', type=int, default=0,
                        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument('--critic_zero_stage', type=int, default=0,
                        help='ZeRO optimization stage for Critic model (and reward).')
    parser.add_argument('--offload', action='store_true', help='Enable ZeRO Offload techniques.')
    parser.add_argument('--offload_reference_model', action='store_true',
                        help='Enable ZeRO Offload techniques for reference model')
    parser.add_argument("--actor_gradient_checkpointing", action="store_true",
                        help="whether to use gradient checkpointing for actor model")
    parser.add_argument("--critic_gradient_checkpointing", action="store_true",
                        help="whether to use gradient checkpointing for critic model")
    parser.add_argument("--unpin_actor_parameters", action='store_true',
                        help="Unpin actor's parameters during generation. This makes generation slower but requires less memory.")
    parser.add_argument("--release_inference_cache", action='store_true',
                        help="Release the memory cache used for inference. This makes generation preparation slower but might increase e2e throughput by using larger batch size.")
    parser.add_argument("--inference_tp_size", type=int, default=1,
                        help="Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature.")
    parser.add_argument("--tp_gather_partition_size", type=int, default=8,
                        help="Granularity to bring in layers for TP sharding inside the hybrid engine. Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature.")
    # parser.add_argument("--num_layers_unfrozen", type=int, default=-1, help="Number of layers to unfreeze for fine-tuning")
    parser.add_argument('--enable_ema', action='store_true', help='Enable EMA checkpoint for the model.')
    # lora
    parser.add_argument("--actor_lora_rank", type=int, default=0)
    parser.add_argument("--actor_lora_alpha", type=int, default=16)
    parser.add_argument("--critic_lora_rank", type=int, default=0)
    parser.add_argument("--critic_lora_alpha", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=1)
    parser.add_argument("--lora_train_bias", type=str, default="none")
    # eval
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--eval_filename", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch",
                        help='- `"no"`: No evaluation is done during training.'
                             '- `"steps"`: Evaluation is done (and logged) every `eval_steps`.'
                             '- `"epoch"`: Evaluation is done at the end of each epoch.')
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)
    # pred
    parser.add_argument("--do_pred", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_filename", type=str, default=None)
    parser.add_argument("--output_filename", type=str, default=None)

    args = parser.parse_args()

    return args


def create_datasets(args, tokenizer_padding_from_left, ppo_ptx_enabled, tokenizer_padding_from_right):
    train_dataset = RLHFDataset(args, os.path.join(args.data_dir, args.train_filename),
                                tokenizer_padding_from_left)
    iters_prompt = len(train_dataset) // args.train_batch_size

    if ppo_ptx_enabled:
        pretrain_dataset = SFTDataset(args, os.path.join(args.data_dir, args.pretrain_filename),
                                      tokenizer_padding_from_right)
        iters_pretrain = len(pretrain_dataset) // args.train_batch_size
    else:
        pretrain_dataset = None
        iters_pretrain = np.inf

    num_update_steps_per_epoch = min(iters_prompt, iters_pretrain) * \
                                 (args.train_batch_size / args.ppo_train_batch_size) * \
                                 args.ppo_epochs / args.gradient_accumulation_steps
    num_total_iters = int(args.num_epochs * num_update_steps_per_epoch)

    return train_dataset, pretrain_dataset, num_total_iters


def create_dataloader(args, train_dataset, pretrain_dataset=None):

    # DataLoaders creation:
    # data_collator = DataCollatorRLHF(args.max_length, pad_token_id)
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(train_dataset)
        if pretrain_dataset is not None:
            pretrain_sampler = RandomSampler(pretrain_dataset)
    else:
        prompt_train_sampler = DistributedSampler(train_dataset)
        if pretrain_dataset is not None:
            pretrain_sampler = DistributedSampler(pretrain_dataset)

    # prompt_train_dataloader is only used for generate_experience() where batch size is fiex at 1
    prompt_train_dataloader = DataLoader(
        train_dataset,
        # collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=1)
        # batch_size=args.train_batch_size)
    if pretrain_dataset is not None:
        pretrain_dataloader = DataLoader(
            pretrain_dataset,
            # collate_fn=default_data_collator,
            sampler=pretrain_sampler,
            batch_size=args.train_batch_size)
    else:
        pretrain_dataloader = [None] * len(
            prompt_train_dataloader)

    # num_update_steps_per_epoch = min(len(prompt_train_dataloader), len(pretrain_dataloader)) * \
    #                              (args.train_batch_size / args.ppo_train_batch_size) * \
    #                              args.ppo_epochs / args.gradient_accumulation_steps
    # num_total_iters = int(args.num_epochs * num_update_steps_per_epoch)

    return prompt_train_dataloader, pretrain_dataloader#, num_total_iters


def main():
    args = get_parser()

    assert not args.enable_hybrid_engine, "DeepSpeed currently does not support Pangu-based or GLM-based model in hybrid engine"

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # deepspeed.init_distributed()

    # args.global_rank = torch.distributed.get_rank()
    if args.local_rank <= 0:
        logger.info(f"Parameters: {args}")

    set_seed(args.seed)
    # torch.distributed.barrier()

    # Set PPO-ptx
    ppo_ptx_enabled = args.pretrain_filename is not None
    if ppo_ptx_enabled:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps
    # n_gpus = torch.distributed.get_world_size()
    n_gpus = torch.cuda.device_count()
    args.global_train_batch_size_actor = args.ppo_train_batch_size * args.gradient_accumulation_steps_actor * n_gpus
    args.global_train_batch_size_critic = args.ppo_train_batch_size * args.gradient_accumulation_steps * n_gpus

    # load tokenizer
    tokenizer_padding_from_right = AutoTokenizer.from_pretrained(args.tokenizer_path, use_cache=False, trust_remote_code=True)
    tokenizer_padding_from_left = copy.deepcopy(tokenizer_padding_from_right)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer_padding_from_left.padding_side = "left" # PS: padding side slightly affect output of sft generation and reward model result
    args.max_prompt_length = args.max_length - args.max_gen_length

    print_gpu_utilization("before create deepspeed rlhf engine", args.local_rank)
    print_gpu_utilization_torch("before create deepspeed rlhf engine", args.local_rank)

    if args.do_train:
        # load data and create dataset
        prompt_train_dataset, pretrain_dataset, num_total_iters = create_datasets(args, tokenizer_padding_from_left,
                                                                                  ppo_ptx_enabled, tokenizer_padding_from_right)
        args.warmup_steps = int(num_total_iters * args.warmup_ratio)

        # load rlhf engine
        rlhf_engine = DeepSpeedRLHFEngine(
            actor_model_name_or_path=args.actor_model_path,
            critic_model_name_or_path=args.critic_model_path,
            tokenizer=tokenizer_padding_from_right,
            num_total_iters=num_total_iters,
            args=args)

        # create dataloader [need to be called after rlhf engine initialization because
        # DistributedSampler can only be called after deepspeed.initialize() is called]
        prompt_train_dataloader, pretrain_dataloader = create_dataloader(args, prompt_train_dataset,
                                                                         pretrain_dataset)

        # create deepspeed ppo trainer
        print_gpu_utilization("after create deepspeed rlhf engine", args.local_rank)
        print_gpu_utilization_torch("after create deepspeed rlhf engine", args.local_rank)
        ppo_trainer = DeepSpeedPPOPTXTrainer if ppo_ptx_enabled else DeepSpeedPPOTrainer
        trainer = ppo_trainer(rlhf_engine, args)
        print_gpu_utilization("after create deepspeed trainer", args.local_rank)
        print_gpu_utilization_torch("after create deepspeed trainer", args.local_rank)

        # create ppo experience dataset
        exp_mini_dataset = PPODataset(args.ppo_batch_numbers,
                                      args.ppo_train_batch_size)
        pretrain_mini_dataset = PPODataset(args.ppo_batch_numbers,
                                           args.ppo_train_batch_size)

        if args.local_rank <= 0:
            logger.info("Start training")

        for epoch in range(args.num_epochs):
            if args.local_rank <= 0:
                logger.info(f"Beginning of Epoch {epoch+1}/{args.num_epochs}, "
                            f"Total Generation Batches {min(len(prompt_train_dataloader), len(pretrain_dataloader))}")
            prompt_iter = iter(prompt_train_dataloader)
            pretrain_iter = iter(pretrain_dataloader)
            step = 0
            # for step, (batch_prompt, batch_pretrain) in enumerate(zip(prompt_train_dataloader, pretrain_dataloader)):
            while True:
                # generate sequence: generate only one sequence at a time, aggregate to form a batch
                answer_start_indices = []
                output_sequences = dict()
                for _ in range(args.train_batch_size):
                    try:
                        batch_prompt = next(prompt_iter)
                        batch_prompt = {k: v.to(device) for k, v in batch_prompt.items()}
                        outputs, prompt_length = trainer.generate_sequence(batch_prompt)
                        answer_start_indices.append(prompt_length-1)
                        for key, val in outputs.items():
                            if key not in output_sequences:
                                output_sequences[key] = []
                            output_sequences[key].append(val[0])
                    except StopIteration:
                        break
                if len(output_sequences) > 0:
                    output_sequences = {key: torch.stack(val) for key, val in output_sequences.items()}
                    output_experiences = trainer.generate_experience(output_sequences, answer_start_indices, device)
                    exp_dataset = exp_mini_dataset.add(output_experiences)
                else:
                    exp_dataset = None

                try:
                    batch_pretrain = next(pretrain_iter)
                    if batch_pretrain is not None:
                        batch_pretrain = {k: v.to(device) for k, v in batch_pretrain.items()}
                        pretrain_dataset = pretrain_mini_dataset.add(batch_pretrain)
                    else:
                        pretrain_dataset = pretrain_mini_dataset.add([[None] * args.train_batch_size])
                except StopIteration:
                    pass

                if exp_dataset is not None:
                    inner_iter = 0
                    critic_loss, actor_loss, pretrain_loss = 0, 0, 0
                    average_reward = 0

                    if args.actor_gradient_checkpointing:
                        rlhf_engine.actor.gradient_checkpointing_enable()

                    for ppo_ep in range(args.ppo_epochs):
                        for i, (exp_data, pretrain_data) in enumerate(
                                zip(exp_dataset, pretrain_dataset)):
                            actor_loss, critic_loss = trainer.train_rlhf(exp_data)
                            critic_loss += actor_loss.item()
                            actor_loss += critic_loss.item()
                            average_reward += exp_data["rewards"].mean()

                            if ppo_ptx_enabled:
                                pretrain_loss = trainer.train_unsupervised(pretrain_data, args.pretrain_coef)
                                pretrain_loss += pretrain_loss.item()

                            inner_iter += 1
                            if args.enable_ema:
                                moving_average(rlhf_engine.actor,
                                               rlhf_engine.actor_ema,
                                               zero_stage=args.actor_zero_stage)

                        random.shuffle(exp_dataset)
                        random.shuffle(pretrain_dataset)

                    if args.local_rank <= 0:
                        logger.info(f'epoch: {epoch}, step: {step}, ppo_ep: {ppo_ep+1}, act_loss: {actor_loss/inner_iter},'
                                    f'cri_loss: {critic_loss/inner_iter}, pretrain_loss: {pretrain_loss/inner_iter}')
                    average_reward = get_all_reduce_mean(average_reward).item()
                    if args.local_rank <= 0:
                        logger.info(f"average reward score: {average_reward/inner_iter}")

                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_disable()

                step += 1

        if args.local_rank <= 0:
            logger.info('saving model ...')

        # if args.actor_lora_rank > 0:
        #     rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
        #     if args.enable_ema:
        #         rlhf_engine.actor_ema = convert_lora_to_linear_layer(rlhf_engine.actor_ema)
        # if args.critic_lora_rank > 0:
        #     rlhf_engine.critic = convert_lora_to_linear_layer(rlhf_engine.critic)

        if args.local_rank == 0:
            save_hf_format(rlhf_engine.actor, tokenizer_padding_from_right, args, sub_folder='actor')
            save_hf_format(rlhf_engine.critic, tokenizer_padding_from_right, args, sub_folder='critic')
            if args.enable_ema:
                save_hf_format(rlhf_engine.actor_ema, tokenizer_padding_from_right, args, sub_folder='actor_ema')

        if args.actor_zero_stage == 3:
            save_zero_three_model(rlhf_engine.actor, global_rank=args.local_rank,
                                  save_dir=os.path.join(args.output_dir, 'actor'),
                                  zero_stage=args.actor_zero_stage)
            if args.enable_ema:
                save_zero_three_model(rlhf_engine.actor_ema, global_rank=args.local_rank,
                                      save_dir=os.path.join(args.output_dir, 'actor_ema'),
                                      zero_stage=args.actor_zero_stage)
        if args.critic_zero_stage == 3:
            save_zero_three_model(rlhf_engine.critic, global_rank=args.local_rank,
                                  save_dir=os.path.join(args.output_dir, 'critic'),
                                  zero_stage=args.critic_zero_stage)


if __name__ == "__main__":
    main()
