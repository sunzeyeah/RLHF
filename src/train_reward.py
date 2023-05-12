
import sys
sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")
sys.path.insert(0, "/mnt/sfevol775196/sunzeye273/Code/chatgpt")
# sys.path.insert(0, "/mnt/share-pa002-vol682688-prd/sunzeye273/Code/chatgpt")
sys.path.insert(0, "/mnt/pa002-28359-vol543625-private/Code/chatgpt")
import glob
import os
import torch
import argparse
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, SequentialSampler

from src.models.reward import RewardModel
from src.utils import logger, RESOURCE_PATH
from src.utils.file_utils import set_seed
from src.data.data import SFTDataset, PairwiseDataset, DataCollatorReward


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
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
    parser.add_argument("--freeze_ratio", type=float, default=0.0, help="ratio of layers frozen for reward training")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=int, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, default="steps",
                        help='- `"no"`: No save is done during training.'
                             '- `"epoch"`: Save is done at the end of each epoch.'
                             '- `"steps"`: Save is done every `save_steps`.')
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--deepspeed_config", type=str, default=None)
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


def main():
    args = get_parser()
    if args.local_rank <= 0:
        logger.info(f"Parameters: {args}")

    set_seed(args.seed)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_cache=False, trust_remote_code=True)

    if "pangu" in args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
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
        layers = reward_model.transformer.h
    elif "chatglm" in args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, trust_remote_code=True).half()
        model.config.lora_rank = args.lora_rank
        model.config.lora_alpha = args.lora_alpha
        model.config.lora_train_bias = args.lora_train_bias
        # Initialize the reward model from the (supervised) fine-tuned SFT model
        reward_model = RewardModel(model.config, model.transformer, tokenizer)
        # reward_model = RewardModelWithLoRA(model.config, model.glm, tokenizer)
        layers = reward_model.transformer.layers
    elif "glm" in args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model.config.lora_rank = args.lora_rank
        model.config.lora_alpha = args.lora_alpha
        model.config.lora_train_bias = args.lora_train_bias
        # Initialize the reward model from the (supervised) fine-tuned SFT model
        reward_model = RewardModel(model.config, model.glm, tokenizer)
        # reward_model = RewardModelWithLoRA(model.config, model.glm, tokenizer)
        layers = reward_model.transformer.transformer.layers
    else:
        raise ValueError(f"Unsupported model name: {args.model_name_or_path}")
    assert model.config.pad_token_id == tokenizer.pad_token_id

    # Freeze the first 70% of the hidden layers of the reward model backbone
    num_layers = len(layers)
    num_frozen = int(args.freeze_ratio * num_layers)
    for layer in layers[:num_frozen]:
        layer.requires_grad_(False)

    if args.checkpoint is not None:
        checkpoints = glob.glob(args.checkpoint.replace("star", "*"))
        st = dict()
        for checkpoint in checkpoints:
            st.update(torch.load(checkpoint, map_location="cpu"))
        res = reward_model.load_state_dict(st, strict=False)
    logger.info(f"Finished loading model and tokenizer")

    # Set up the datasets
    if args.do_train:
        train_dataset = PairwiseDataset(args, os.path.join(args.data_dir, args.train_filename),
                                        tokenizer)
    else:
        train_dataset = None
    if args.do_eval:
        val_dataset = PairwiseDataset(args, os.path.join(args.data_dir, args.eval_filename),
                                      tokenizer)
    else:
        val_dataset = None
    if args.do_pred:
        test_dataset = SFTDataset(args, os.path.join(args.data_dir, args.test_filename),
                                  tokenizer)
    else:
        test_dataset = None

    # training arguments
    deepspeed_config = os.path.join(RESOURCE_PATH, "config", "deepspeed", args.deepspeed_config) if args.deepspeed_config is not None else None
    if torch.cuda.is_available():
        bf16 = torch.cuda.get_device_capability()[0] >= 8
        fp16 = False if bf16 else True
    else:
        fp16 = False
        bf16 = False
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        no_cuda=not torch.cuda.is_available(),
        seed=args.seed,
        data_seed=args.seed,
        local_rank=args.local_rank,
        do_train=args.do_train,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.train_batch_size,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        half_precision_backend="auto",
        fp16=fp16,
        bf16=bf16,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        deepspeed=deepspeed_config,
        gradient_checkpointing=args.gradient_checkpointing,
        do_eval=args.do_eval,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        per_device_eval_batch_size=args.eval_batch_size,
        do_predict=args.do_pred,
        use_legacy_prediction_loop=args.do_pred,
    )
    if args.local_rank <= 0:
        logger.info(f"Training Arguments: {training_args}")

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
        # data_collator=data_collator,
    )

    if args.do_train:
        trainer.train()
        trainer.save_model(args.output_dir)

    elif args.do_eval:
        eval_result = trainer.evaluate(eval_dataset=val_dataset)
        logger.info(eval_result)

    if args.do_pred:
        model.eval()
        device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
        reward_model = reward_model.half().to(device)
        sampler = SequentialSampler(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, sampler=sampler)
        rewards = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Prediction"):
                chosen_input_ids = batch['input_ids'].to(device)
                chosen_attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
                chosen_position_ids = batch['position_ids'].to(device) if 'position_ids' in batch else None
                output = reward_model(chosen_input_ids, chosen_attention_mask, chosen_position_ids)
                rewards.extend(output['chosen_reward'].cpu().detach().tolist())
        # save result into file
        with open(os.path.join(args.output_dir, args.output_filename), "w", encoding="utf-8") as w:
            w.write("\t".join(("prompt", "answer", "score"))+"\n")
            for item, reward in zip(test_dataset.post_list, rewards):
                w.write("\t".join((item["prompt"], item["label"], str(reward))) + "\n")
        logger.info(f"Finished prediction and saving into {args.output_filename}")


if __name__ == "__main__":
    main()
