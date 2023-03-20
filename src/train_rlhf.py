
import sys
sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")
sys.path.insert(0, "/mnt/private-pa002-vol726121-prd/Code/RLHF")

import os
import argparse
import torch
import trlx

from trlx.data.configs import TRLConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils import logger, RESOURCE_PATH
from src.models.reward import GPTRewardModel
from src.utils.file_utils import set_seed
from src.utils.data import RLHFDataset


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--sft_checkpoint", type=str, required=True)
    parser.add_argument("--reward_checkpoint", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=1024)
    # train
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--train_filename", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        help="transformers.trainer_utils.SchedulerType, including:"
                             "linear, cosine, cosine_with_restarts, polynomial, constant,"
                             "constant_with_warmup")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=int, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        help='- `"no"`: No save is done during training.'
                             '- `"epoch"`: Save is done at the end of each epoch.'
                             '- `"steps"`: Save is done every `save_steps`.')
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", type=bool, default=False,
                        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--ppo_config", type=str, default=None)
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


def main():
    args = get_parser()
    logger.info(f"Parameters: {args}")

    set_seed(args.seed)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)

    # load reward model
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
    model.resize_token_embeddings(tokenizer.vocab_size)
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    # assert rw_tokenizer.pad_token_id == rw_tokenizer.eos_token_id
    # rw_model.config.end_token_id = rw_tokenizer.eos_token_id
    # rw_model.config.pad_token_id = rw_model.config.eos_token_id
    rw_model = GPTRewardModel(model, tokenizer)
    state_dict_reward = torch.load(args.reward_checkpoint, map_location="cpu")
    rw_model.load_state_dict(state_dict_reward)
    logger.info(f"Finish loading reward model from {args.reward_checkpoint}")

    rw_model.half()
    rw_model.eval()
    device = torch.device(f"cuda:{args.local_rank}")
    rw_model.to(device)

    def get_scores(samples):
        scores_list = []
        batch_size = 2
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i: i + batch_size]
            # sub_samples = [tokenizer.bos_token + chosen + tokenizer.eos_token for chosen in sub_samples]
            encodings_dict = tokenizer(
                sub_samples,
                max_length=ppo_config.train.seq_length,
                truncation="longest_first",
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(device)
            attn_masks = encodings_dict["attention_mask"].to(device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            scores_list.append(sub_scores["chosen_end_scores"])
        scores = torch.cat(scores_list, dim=0)

        return scores

    def reward_fn(samples, **kwargs):
        # rank = torch.distributed.get_rank()
        # if rank == 0:
        #     logger.info(f"[rank-{rank}]: {samples[0]}")
        original_samples = [text.split("模型回答:")[0] + "模型回答:" for text in samples]
        original_samples = [text + post_summary_dict[text.strip()] for text in original_samples]
        original_scores = get_scores(original_samples)
        scores = get_scores(samples)
        norms_scores = scores - original_scores
        return norms_scores

    # config_path = pathlib.Path(__file__).parent.joinpath("configs/ppo_config_pangu.yml")
    ppo_config = TRLConfig.load_yaml(os.path.join(RESOURCE_PATH, "config", "ppo_config", args.ppo_config))
    # TODO: 待确定trlx中如何更新sft模型
    ppo_config.model.model_path = args.sft_checkpoint
    # config.tokenizer.tokenizer_path = GPT_Token_PATH
    # tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    logger.info(f"PPO config: {ppo_config}")
    max_length = ppo_config.train.seq_length - ppo_config.method.gen_kwargs["max_new_tokens"]

    # load dataset
    post_summary_dict = dict()
    if args.do_train:
        train_data = RLHFDataset.load_dataset(os.path.join(args.data_dir, args.train_filename), max_length)
        for td in train_data:
            post_summary_dict[td['prompt']] = td['label']
        train_dataset = RLHFDataset(train_data, tokenizer, max_length=max_length)
    else:
        train_dataset = None
    if args.do_eval:
        dev_data = RLHFDataset.load_dataset(os.path.join(args.data_dir, args.eval_filename), max_length)
        for td in dev_data:
            post_summary_dict[td['prompt']] = td['label']
        dev_dataset = RLHFDataset(dev_data, tokenizer, max_length=max_length)
    else:
        dev_dataset = None

    if args.do_train:
        trlx.train(
            reward_fn=reward_fn,
            prompts=train_dataset,
            eval_prompts=dev_dataset,#[0:1000],  # sampling 1000 validation prompts for evaluation speed in training
            config=ppo_config,
        )


if __name__ == "__main__":
    main()
