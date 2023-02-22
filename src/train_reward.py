
import sys
sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")

import os
import torch
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

from src.models.reward import GPTRewardModel
from src.utils import logger, RESOURCE_PATH
from src.utils.file_utils import set_seed
from src.utils.data import PairwiseDataset, DataCollatorReward


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)

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
    parser.add_argument("--save_total_limit", type=int, default=20)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", type=bool, default=False,
                        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--deepspeed_config", type=str, default=None)
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

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
    tokenizer.add_special_tokens({'unk_token': "<unk>",
                                  'bos_token': "<s>",
                                  'eos_token': "<eot>",
                                  'pad_token': "<pad>"})

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer.sp))
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # training arguments
    deepspeed_config = os.path.join(RESOURCE_PATH, "config", "reward_model", args.deepspeed_config) if args.deepspeed_config is not None else None
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        no_cuda=not torch.cuda.is_available(),
        seed=args.seed,
        data_seed=args.seed,
        # local_rank=args.local_rank,
        logging_steps=args.logging_steps,
        do_train=args.do_train,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        half_precision_backend="auto",
        fp16=torch.cuda.is_available(),
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        # deepspeed=deepspeed_config,
        do_eval=args.do_eval,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        per_device_eval_batch_size=args.eval_batch_size,
        do_predict=args.do_pred,
        use_legacy_prediction_loop=args.do_pred,
    )
    logger.info(f"Training Arguments: {training_args}")

    # Initialize the reward model from the (supervised) fine-tuned SFT model
    reward_model = GPTRewardModel(model, tokenizer)
    if args.checkpoint is not None:
        st = torch.load(args.checkpoint, map_location="cpu")
        reward_model.load_state_dict(st)
    logger.info(f"Finished loading model and tokenizer")

    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = reward_model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    # Set up the datasets
    if args.do_train:
        train_dataset = PairwiseDataset(os.path.join(args.data_dir, args.train_filename),
                                        tokenizer, max_length=args.max_length)
    else:
        train_dataset = None
    if args.do_eval:
        val_dataset = PairwiseDataset(os.path.join(args.data_dir, args.eval_filename),
                                      tokenizer, max_length=args.max_length)
    else:
        val_dataset = None

    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward()

    def compute_metrics(eval_preds):
        chosen_end_scores = eval_preds.predictions[0]  # chosen scores
        rejected_end_scores = eval_preds.predictions[1]  # rejected scores

        result = {}
        acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
        result["accuracy"] = acc

        return result

    # Prepare the trainer and start training
    trainer = Trainer(
        model=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    if args.do_train:
        trainer.train()
        # trainer.save_model(args.output_dir)

    elif args.do_eval:
        trainer.evaluate(eval_dataset=val_dataset)


if __name__ == "__main__":
    main()
