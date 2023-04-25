
import sys
sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")
sys.path.insert(0, "/mnt/sfevol775196/sunzeye273/Code/chatgpt")
# sys.path.insert(0, "/mnt/share-pa002-vol682688-prd/sunzeye273/Code/chatgpt")
sys.path.insert(0, "/mnt/pa002-28359-vol543625-private/Code/chatgpt")

import os
import argparse
import torch
import random
import deepspeed

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, default_data_collator
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader

from src.utils import logger, RESOURCE_PATH
from src.utils.config import TRLConfig, default_ilql_config, default_ppo_config, default_sft_config
from src.models.rlhf_engine import DeepSpeedRLHFEngine
from src.models.trainer import DeepSpeedPPOTrainer, DeepSpeedPPOPTXTrainer
from src.utils.file_utils import set_seed
from src.data.data import RLHFDataset, SFTDataset, DataCollatorRLHF
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
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_gen_length", type=int, default=256)
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
    parser.add_argument("--critic_lora_rank", type=int, default=0)
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


def create_datasets(args, ppo_ptx_enabled):
    train_dataset = SFTDataset.load_dataset(os.path.join(args.data_dir, args.train_filename))
    if ppo_ptx_enabled:
        pretrain_dataset = SFTDataset.load_dataset(os.path.join(args.data_dir, args.pretrain_filename))

    # DataLoaders creation:
    data_collator = DataCollatorRLHF(args.max_length, args.inference_tp_size)
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(train_dataset)
        if ppo_ptx_enabled:
            pretrain_sampler = RandomSampler(pretrain_dataset)
    else:
        prompt_train_sampler = DistributedSampler(train_dataset)
        if ppo_ptx_enabled:
            pretrain_sampler = DistributedSampler(pretrain_dataset)

    prompt_train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.train_batch_size)
    if ppo_ptx_enabled:
        pretrain_dataloader = DataLoader(
            pretrain_dataset,
            collate_fn=default_data_collator,
            sampler=pretrain_sampler,
            batch_size=args.train_batch_size)
    else:
        pretrain_dataloader = [None] * len(
            prompt_train_dataloader)  # basically a dummy dataloader

    num_update_steps_per_epoch = min(len(prompt_train_dataloader), len(pretrain_dataloader)) * \
                                 (args.train_batch_size / args.ppo_train_batch_size) * \
                                 args.ppo_epochs / args.gradient_accumulation_steps
    num_total_iters = int(args.num_epochs * num_update_steps_per_epoch)

    return prompt_train_dataloader, pretrain_dataloader, num_total_iters


def main():
    args = get_parser()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    if args.global_rank <= 0:
        logger.info(f"Parameters: {args}")

    set_seed(args.seed)
    torch.distributed.barrier()

    # Set PPO-ptx
    ppo_ptx_enabled = args.pretrain_filename is not None
    if ppo_ptx_enabled:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps
    n_gpus = torch.distributed.get_world_size()
    args.global_train_batch_size_actor = args.ppo_train_batch_size * args.gradient_accumulation_steps_actor * n_gpus
    args.global_train_batch_size_critic = args.ppo_train_batch_size * args.gradient_accumulation_steps * n_gpus

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_cache=False, trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left" # PS: padding side does affect output of reward model

    # # load reward model
    # if "pangu" in args.reward_model_path:
    #     model = AutoModelForCausalLM.from_pretrained(args.reward_model_path, use_cache=False, trust_remote_code=True)
    #     model.resize_token_embeddings(tokenizer.vocab_size)
    #     # model.config.end_token_id = tokenizer.eos_token_id
    #     # model.config.pad_token_id = tokenizer.pad_token_id
    #     # model.config.bos_token_id = tokenizer.bos_token_id
    #     # model.config.eos_token_id = tokenizer.eos_token_id
    #     model.config.lora_rank = args.lora_rank
    #     model.config.lora_alpha = args.lora_alpha
    #     model.config.lora_train_bias = args.lora_train_bias
    #     # Initialize the reward model from the (supervised) fine-tuned SFT model
    #     reward_model = RewardModel(model.config, model.transformer, tokenizer)
    #     # reward_model = RewardModelWithLoRA(model.config, model.transformer, tokenizer)
    # elif "chatglm" in args.reward_model_path:
    #     model = AutoModelForSeq2SeqLM.from_pretrained(args.reward_model_path, trust_remote_code=True).half()
    #     model.config.lora_rank = args.lora_rank
    #     model.config.lora_alpha = args.lora_alpha
    #     model.config.lora_train_bias = args.lora_train_bias
    #     # Initialize the reward model from the (supervised) fine-tuned SFT model
    #     reward_model = RewardModel(model.config, model, tokenizer)
    #     # reward_model = RewardModelWithLoRA(model.config, model.glm, tokenizer)
    # elif "glm" in args.reward_model_path:
    #     model = AutoModelForSeq2SeqLM.from_pretrained(args.reward_model_path, trust_remote_code=True)
    #     model.config.lora_rank = args.lora_rank
    #     model.config.lora_alpha = args.lora_alpha
    #     model.config.lora_train_bias = args.lora_train_bias
    #     # Initialize the reward model from the (supervised) fine-tuned SFT model
    #     reward_model = RewardModel(model.config, model.glm, tokenizer)
    #     # reward_model = RewardModelWithLoRA(model.config, model.glm, tokenizer)
    # else:
    #     raise ValueError(f"Unsupported model name: {args.reward_model_path}")
    # assert model.config.pad_token_id == tokenizer.pad_token_id
    #
    # if args.reward_checkpoint is not None:
    #     checkpoints = glob.glob(args.reward_checkpoint.replace("star", "*"))
    #     st = dict()
    #     for checkpoint in checkpoints:
    #         st.update(torch.load(checkpoint, map_location="cpu"))
    #     res = reward_model.load_state_dict(st, strict=False)
    #
    # device = torch.device(f"cuda:{args.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    # # reward_model.half()
    # reward_model.eval()
    # reward_model.to(device)
    # logger.info(f"Finish loading reward model from {args.reward_checkpoint}")
    #
    # def reward_fn(samples, **kwargs):
    #     scores_list = []
    #     for i in range(0, len(samples), ppo_config.train.batch_size):
    #         input_ids_list = []
    #         attention_mask_list = []
    #         position_ids_list = []
    #         for sample in samples[i: i + ppo_config.train.batch_size]:
    #             prompt, pred = sample.split(tokenizer.sep_token, maxsplit=1)
    #             logger.debug(f"prompt: {prompt}, pred: {pred}")
    #             if "pangu" in ppo_config.model.model_path:
    #                 encodings_dict = tokenizer(prompt, pred, max_length=ppo_config.train.seq_length,
    #                                            truncation="longest_first", padding="max_length", return_tensors="pt",
    #                                            return_token_type_ids=False)
    #                 input_ids_list.append(encodings_dict["input_ids"])
    #                 attention_mask_list.append(encodings_dict["attention_mask"])
    #             elif "chatglm" in ppo_config.model.model_path:
    #                 encoded_dict = tokenizer(prompt, pred, max_length=ppo_config.train.seq_length, return_tensors="pt",
    #                                          truncation="longest_first", padding="max_length")
    #                 input_ids_list.append(encoded_dict["input_ids"][0])
    #             elif "glm" in ppo_config.model.model_path:
    #                 # TODO: to be modified for and tested against glm
    #                 encoded_prompt = tokenizer(prompt, tokenizer.mask_token)
    #                 prompt_length = len(encoded_prompt['input_ids'])
    #                 label_length = len(tokenizer.tokenize(pred))
    #                 if prompt_length + label_length > ppo_config.train.seq_length:
    #                     num_tokens_to_remove = prompt_length + label_length - ppo_config.train.seq_length
    #                     for _ in range(num_tokens_to_remove):
    #                         if prompt_length > label_length:
    #                             prompt_length -= 1
    #                         else:
    #                             label_length -= 1
    #                 else:
    #                     label_length = ppo_config.train.seq_length - prompt_length
    #                 assert prompt_length > 0
    #                 assert label_length > 0
    #                 assert prompt_length + label_length <= ppo_config.train.seq_length
    #                 encoded_dict = tokenizer(prompt, tokenizer.mask_token,
    #                                          max_length=prompt_length, truncation="only_first",
    #                                          return_tensors="pt", return_attention_mask=True,
    #                                          return_token_type_ids=False)
    #                 encoded_dict = tokenizer.build_inputs_for_generation(encoded_dict, targets=pred,
    #                                                                      max_gen_length=label_length, padding=True)
    #                 input_ids_list.append(encoded_dict["input_ids"][0])
    #                 attention_mask_list.append(encoded_dict["attention_mask"][0])
    #                 position_ids_list.append(encoded_dict["position_ids"][0])
    #             else:
    #                 raise ValueError(f"Unsupported model type: {ppo_config.model.model_path}")
    #         # encodings_dict = tokenizer(
    #         #     sub_samples,
    #         #     max_length=ppo_config.train.seq_length,
    #         #     truncation="longest_first",
    #         #     padding="max_length",
    #         #     return_tensors="pt",
    #         # )
    #         input_ids = torch.stack(input_ids_list, dim=0).to(device)
    #         attention_mask = torch.stack(attention_mask_list, dim=0).to(device) if len(attention_mask_list) > 0 else None
    #         position_ids = torch.stack(position_ids_list, dim=0).to(device) if len(position_ids_list) > 0 else None
    #         with torch.no_grad():
    #             sub_scores = reward_model(input_ids, attention_mask, position_ids)
    #         scores_list.append(sub_scores["chosen_reward"])
    #
    #     scores = torch.cat(scores_list, dim=0)
    #
    #     return scores
    #
    # # load ppo config
    # ppo_config = TRLConfig.load_yaml(os.path.join(RESOURCE_PATH, "config", "ppo_model", args.ppo_config))
    # ppo_config.train.epochs = args.num_epochs
    # ppo_config.train.seq_length = args.max_length
    # ppo_config.train.batch_size = args.train_batch_size
    # ppo_config.train.checkpoint_dir = args.output_dir
    # ppo_config.train.checkpoint_interval = args.save_steps
    # ppo_config.train.eval_interval = args.eval_steps
    # ppo_config.model.num_layers_unfrozen = args.num_layers_unfrozen
    # ppo_config.model.model_path = args.sft_model_path
    # ppo_config.tokenizer.tokenizer_path = args.tokenizer_path
    # ppo_config.optimizer.kwargs['lr'] = args.learning_rate
    # ppo_config.optimizer.kwargs['weight_decay'] = args.weight_decay
    # ppo_config.method.chunk_size = args.eval_batch_size
    # ppo_config.train.lora_rank = args.lora_rank
    # ppo_config.train.lora_alpha = args.lora_alpha
    # ppo_config.train.lora_train_bias = args.lora_train_bias
    # logger.info(f"PPO config: {ppo_config}")

    # load dataset
    if args.do_train:
        prompt_train_dataloader, pretrain_dataloader, num_total_iters = create_datasets(args, ppo_ptx_enabled)
        args.warmup_steps = int(num_total_iters * args.warmup_ratio)
    # if args.do_eval:
    #     dev_dataset = SFTDataset.load_dataset(os.path.join(args.data_dir, args.eval_filename))
    # else:
    #     dev_dataset = None

    if args.do_train:
        rlhf_engine = DeepSpeedRLHFEngine(
            actor_model_name_or_path=args.actor_model_path,
            critic_model_name_or_path=args.critic_model_path,
            tokenizer=tokenizer,
            num_total_iters=num_total_iters,
            args=args)

        ppo_trainer = DeepSpeedPPOPTXTrainer if ppo_ptx_enabled else DeepSpeedPPOTrainer
        trainer = ppo_trainer(rlhf_engine, args)

        exp_mini_dataset = RLHFDataset(args.ppo_batch_numbers,
                                       args.ppo_train_batch_size)
        pretraib_mini_dataset = RLHFDataset(args.ppo_batch_numbers,
                                         args.ppo_train_batch_size)

        if args.global_rank <= 0:
            logger.info("Start training")

        for epoch in range(args.num_epochs):
            if args.global_rank <= 0:
                logger.info(f"Beginning of Epoch {epoch+1}/{args.num_epochs}, "
                            f"Total Generation Batches {min(len(prompt_train_dataloader), len(pretrain_dataloader))}")
            for step, (batch_prompt, batch_pretrain) in enumerate(zip(prompt_train_dataloader, pretrain_dataloader)):
                batch_prompt = {k: v.to(device) for k, v in batch_prompt.items()}
                # for k, v in batch_prompt.items():
                #     try:
                #         batch_prompt[k] = v.to(device)
                #     except:
                #         batch_prompt[k] = v
                if batch_pretrain is not None:
                    batch_pretrain = {k: v.to(device) for k, v in batch_pretrain.items()}
                    pretrain_dataset = pretraib_mini_dataset.add(batch_pretrain)
                else:
                    pretrain_dataset = pretraib_mini_dataset.add([[None] * args.train_batch_size])
                prompts = batch_prompt['prompt']
                length = prompts.size(-1)
                if length > args.max_length:
                    prompts = prompts[:, length - args.max_length:]
                    raise ValueError("Prompt length is too long")

                out = trainer.generate_experience(prompts)
                exp_dataset = exp_mini_dataset.add(out)

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

                    if args.global_rank <= 0:
                        logger.info(f'epoch: {epoch}, step: {step}, ppo_ep: {ppo_ep+1}, act_loss: {actor_loss/inner_iter},'
                                    f'cri_loss: {critic_loss/inner_iter}, pretrain_loss: {pretrain_loss/inner_iter}')
                    average_reward = get_all_reduce_mean(average_reward).item()
                    if args.global_rank <= 0:
                        logger.info(f"average reward score: {average_reward/inner_iter}")

                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_disable()

        if args.global_rank <= 0:
            logger.info('saving model ...')

        if args.actor_lora_rank > 0:
            rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
            if args.enable_ema:
                rlhf_engine.actor_ema = convert_lora_to_linear_layer(
                    rlhf_engine.actor_ema)
        if args.critic_lora_rank > 0:
            rlhf_engine.critic = convert_lora_to_linear_layer(rlhf_engine.critic)

        if torch.distributed.get_rank() == 0:
            save_hf_format(rlhf_engine.actor, tokenizer, args, sub_folder='actor')
            save_hf_format(rlhf_engine.critic, tokenizer, args, sub_folder='critic')
            if args.enable_ema:
                save_hf_format(rlhf_engine.actor_ema, tokenizer, args, sub_folder='actor_ema')

        if args.actor_zero_stage == 3:
            save_zero_three_model(rlhf_engine.actor, global_rank=args.global_rank,
                                  save_dir=os.path.join(args.output_dir, 'actor'),
                                  zero_stage=args.actor_zero_stage)
            if args.enable_ema:
                save_zero_three_model(rlhf_engine.actor_ema, global_rank=args.global_rank,
                                      save_dir=os.path.join(args.output_dir, 'actor_ema'),
                                      zero_stage=args.actor_zero_stage)
        if args.critic_zero_stage == 3:
            save_zero_three_model(rlhf_engine.critic, global_rank=args.global_rank,
                                  save_dir=os.path.join(args.output_dir, 'critic'),
                                  zero_stage=args.critic_zero_stage)


if __name__ == "__main__":
    main()
