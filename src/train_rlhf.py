
import sys
sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")
sys.path.insert(0, "/mnt/sfevol775196/sunzeye273/Code/RLHF")
sys.path.insert(0, "/mnt/pa002-28359-vol543625-private/Code/RLHF")

import os
import argparse
import torch
import glob

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from src.utils import logger, RESOURCE_PATH
from src.utils.config import TRLConfig, default_ilql_config, default_ppo_config, default_sft_config
from src.models.reward import RewardModel
from src.utils.file_utils import set_seed
from src.data.data import RLHFDataset
from src.utils.loading import get_pipeline, get_trainer


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--sft_model_path", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True)
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
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--ppo_config", type=str, default=None)
    parser.add_argument("--num_layers_unfrozen", type=int, default=-1, help="Number of layers to unfreeze for fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=0)
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


def train(model_path: Optional[str] = None,
          reward_fn: Optional[Callable[[List[Any], Any], torch.Tensor]] = None,
          dataset: Optional[Iterable[Tuple[str, float]]] = None,
          samples: Optional[List[str]] = None,
          rewards: Optional[List[float]] = None,
          prompts: Optional[List[str]] = None,
          eval_prompts: Optional[List[str]] = None,
          metric_fn: Optional[Callable[[List[str], List[str], List[str]], Dict[str, List[float]]]] = None,
          config: Optional[TRLConfig] = None,
          stop_sequences: Optional[List[str]] = [],):
    if config is None:
        logger.warn(
            "Passing the `config` argument implicitly is depreciated, use or"
            "adapt some from default configs instead"
        )
        if reward_fn:
            config = default_ppo_config()
        elif rewards:
            config = default_ilql_config()
        else:
            config = default_sft_config()

    set_seed(config.train.seed)

    if dataset:
        logger.warn("the `dataset` argument is being depreciated, split it into `samples` and `rewards` instead")
        samples, rewards = dataset

    if model_path:
        config.model.model_path = model_path

    trainer = get_trainer(config.train.trainer)(
        config=config,
        reward_fn=reward_fn,
        metric_fn=metric_fn,
        stop_sequences=stop_sequences,
        **config.train.trainer_kwargs,
    )

    batch_size = config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
    max_prompt_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    # Online training against a reward function (e.g. PPO)
    if reward_fn:
        prompts = prompts or [trainer.tokenizer.bos_token] * batch_size

        if eval_prompts is None:
            eval_prompts = prompts[:batch_size]

        pipeline = get_pipeline(config.train.pipeline)(prompts, config, trainer.tokenizer)
        trainer.add_prompt_pipeline(pipeline)

        if eval_prompts is None:
            eval_prompts = prompts[:batch_size]

        trainer.make_experience(config.method.num_rollouts)

    # Offline training from the collected samples (e.g. SFT, ILQL)
    elif samples:
        if rewards:
            if len(samples) != len(rewards):
                raise ValueError(f"Number of samples {len(samples)} should match the number of rewards {len(rewards)}")

        if eval_prompts is None:
            eval_prompts = [trainer.tokenizer.bos_token] * batch_size

        if rewards:
            trainer.make_experience(samples, rewards, config.train.seq_length)
        else:
            trainer.store = get_pipeline(config.train.pipeline)(samples, config, trainer.tokenizer)

    else:
        raise ValueError("Either `samples` or `reward_fn` should be given for training")

    eval_pipeline = get_pipeline(config.train.pipeline)(eval_prompts, config, trainer.tokenizer)
    trainer.add_eval_pipeline(eval_pipeline)

    trainer.learn()


def main():
    args = get_parser()
    logger.info(f"Parameters: {args}")

    set_seed(args.seed)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_cache=False, trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # PS: padding side does affect output of reward model

    # load reward model
    if "pangu" in args.reward_model_path:
        model = AutoModelForCausalLM.from_pretrained(args.reward_model_path, use_cache=False, trust_remote_code=True)
        model.resize_token_embeddings(tokenizer.vocab_size)
        # model.config.end_token_id = tokenizer.eos_token_id
        # model.config.pad_token_id = tokenizer.pad_token_id
        # model.config.bos_token_id = tokenizer.bos_token_id
        # model.config.eos_token_id = tokenizer.eos_token_id
        model.config.lora_rank = args.lora_rank
        model.config.lora_alpha = args.lora_alpha
        model.config.lora_train_bias = args.lora_train_bias
        # Initialize the reward model from the (supervised) fine-tuned SFT model
        reward_model = RewardModel(model.config, model.transformer, tokenizer)
        # reward_model = RewardModelWithLoRA(model.config, model.transformer, tokenizer)
    elif "glm" in args.reward_model_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.reward_model_path, trust_remote_code=True)
        if "chatglm" in args.reward_model_path:
            model = model.half()
        model.config.lora_rank = args.lora_rank
        model.config.lora_alpha = args.lora_alpha
        model.config.lora_train_bias = args.lora_train_bias
        # Initialize the reward model from the (supervised) fine-tuned SFT model
        reward_model = RewardModel(model.config, model.glm, tokenizer)
        # reward_model = RewardModelWithLoRA(model.config, model.glm, tokenizer)
    else:
        raise ValueError(f"Unsupported model name: {args.reward_model_path}")
    assert model.config.pad_token_id == tokenizer.pad_token_id

    if args.reward_checkpoint is not None:
        checkpoints = glob.glob(args.reward_checkpoint.replace("star", "*"))
        st = dict()
        for checkpoint in checkpoints:
            st.update(torch.load(checkpoint, map_location="cpu"))
        res = reward_model.load_state_dict(st, strict=False)

    device = torch.device(f"cuda:{args.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    reward_model.half()
    reward_model.eval()
    reward_model.to(device)
    logger.info(f"Finish loading reward model from {args.reward_checkpoint}")

    # define reward functions in ppo training
    def get_scores(samples):
        scores_list = []
        for i in range(0, len(samples), ppo_config.train.batch_size):
            input_ids_list = []
            attention_mask_list = []
            position_ids_list = []
            for sample in samples[i: i + ppo_config.train.batch_size]:
                prompt, pred = sample.split(tokenizer.sep_token, maxsplit=1)
                logger.debug(f"prompt: {prompt}, pred: {pred}")
                if "pangu" in ppo_config.model.model_path:
                    encodings_dict = tokenizer(prompt, pred, max_length=ppo_config.train.seq_length,
                                               truncation="longest_first", padding="max_length", return_tensors="pt",
                                               return_token_type_ids=False)
                    input_ids_list.append(encodings_dict["input_ids"])
                    attention_mask_list.append(encodings_dict["attention_mask"])
                elif "glm" in ppo_config.model.model_path:
                    # TODO: to be modified for and tested against glm
                    encoded_prompt = tokenizer(prompt, tokenizer.mask_token)
                    prompt_length = len(encoded_prompt['input_ids'])
                    label_length = len(tokenizer.tokenize(pred)) + (1 if "chatglm" not in ppo_config.model.model_path else 0)
                    if prompt_length + label_length > ppo_config.train.seq_length:
                        num_tokens_to_remove = prompt_length + label_length - ppo_config.train.seq_length
                        for _ in range(num_tokens_to_remove):
                            if prompt_length > label_length:
                                prompt_length -= 1
                            else:
                                label_length -= 1
                    else:
                        label_length = ppo_config.train.seq_length - prompt_length
                    assert prompt_length > 0
                    assert label_length > 0
                    assert prompt_length + label_length <= ppo_config.train.seq_length
                    encoded_dict = tokenizer(prompt, tokenizer.mask_token,
                                             max_length=prompt_length, truncation="only_first",
                                             return_tensors="pt", return_attention_mask=True,
                                             return_token_type_ids=False)
                    encoded_dict = tokenizer.build_inputs_for_generation(encoded_dict, targets=pred,
                                                                         max_gen_length=label_length, padding=True)
                    input_ids_list.append(encoded_dict["input_ids"][0])
                    attention_mask_list.append(encoded_dict["attention_mask"][0])
                    position_ids_list.append(encoded_dict["position_ids"][0])
                else:
                    raise ValueError(f"Unsupported model type: {ppo_config.model.model_path}")
            # encodings_dict = tokenizer(
            #     sub_samples,
            #     max_length=ppo_config.train.seq_length,
            #     truncation="longest_first",
            #     padding="max_length",
            #     return_tensors="pt",
            # )
            input_ids = torch.stack(input_ids_list, dim=0).to(device)
            attention_mask = torch.stack(attention_mask_list, dim=0).to(device)
            position_ids = torch.stack(position_ids_list, dim=0).to(device) if len(position_ids_list) > 0 else None
            with torch.no_grad():
                sub_scores = reward_model(input_ids, attention_mask, position_ids)
            scores_list.append(sub_scores["chosen_reward"])
        scores = torch.cat(scores_list, dim=0)

        return scores

    def reward_fn(samples, **kwargs):
        original_scores = get_scores(samples)
        scores = get_scores(samples)
        norms_scores = scores - original_scores
        return norms_scores

    # load ppo config
    ppo_config = TRLConfig.load_yaml(os.path.join(RESOURCE_PATH, "config", "ppo_model", args.ppo_config))
    ppo_config.train.epochs = args.num_epochs
    ppo_config.train.seq_length = args.max_length
    ppo_config.train.batch_size = args.train_batch_size
    ppo_config.train.checkpoint_interval = args.save_steps
    ppo_config.train.eval_interval = args.eval_steps
    ppo_config.model.num_layers_unfrozen = args.num_layers_unfrozen
    ppo_config.model.model_path = args.sft_model_path
    ppo_config.tokenizer.tokenizer_path = args.tokenizer_path
    ppo_config.optimizer.kwargs['lr'] = args.learning_rate
    ppo_config.optimizer.kwargs['weight_decay'] = args.weight_decay
    ppo_config.method.chunk_size = args.train_batch_size
    ppo_config.train.lora_rank = args.lora_rank
    ppo_config.train.lora_alpha = args.lora_alpha
    ppo_config.train.lora_train_bias = args.lora_train_bias
    logger.info(f"PPO config: {ppo_config}")

    # load dataset
    if args.do_train:
        train_dataset = RLHFDataset.load_dataset(os.path.join(args.data_dir, args.train_filename))
    else:
        train_dataset = None
    if args.do_eval:
        dev_dataset = RLHFDataset.load_dataset(os.path.join(args.data_dir, args.eval_filename))
    else:
        dev_dataset = None

    if args.do_train:
        train(model_path=args.sft_model_path, reward_fn=reward_fn, prompts=train_dataset,
              eval_prompts=dev_dataset, config=ppo_config)


if __name__ == "__main__":
    main()
