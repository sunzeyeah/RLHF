
import sys
sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")

import os
import random
import argparse
import evaluate
import numpy as np
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from src.utils import logger
from src.utils import TLDRDataset


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


# Create a preprocessing function to extract out the proper logits from the model output
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]

    return logits.argmax(dim=-1)


def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    # train
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--train_filename", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        help='- `"no"`: No save is done during training.'
                             '- `"epoch"`: Save is done at the end of each epoch.'
                             '- `"steps"`: Save is done every `save_steps`.')
    parser.add_argument("--save_steps", type=int, default=None)
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
    
    args = parser.parse_args()
    
    return args


def main():
    args = get_parser()

    set_seed(args.seed)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
    tokenizer.add_special_tokens({'eos_token': "<eot>", 'pad_token': "<eot>"})
    # tokenizer.add_special_tokens({'bos_token': "<s>"})
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer.sp))
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    logger.info(f"Finished loading model and tokenizer")

    # Set up the datasets
    if args.do_train:
        train_dataset = TLDRDataset(os.path.join(args.data_dir, args.train_filename),
                                    tokenizer, "train", max_length=args.max_length)
    else:
        train_dataset = None
    if args.do_eval:
        dev_dataset = TLDRDataset(os.path.join(args.data_dir, args.eval_filename),
                                  tokenizer, "valid", max_length=args.max_length)
    else:
        dev_dataset = None

    # Prepare the trainer and start training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=args.do_train,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        warmup_steps=args.warmup_steps,
        half_precision_backend="auto",
        fp16=torch.cuda.is_available(),
        adam_beta1=0.9,
        adam_beta2=0.95,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        deepspeed=args.deepspeed_config,
        do_eval=args.do_eval,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        per_device_eval_batch_size=args.eval_batch_size
    )

    # Set up the metric
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        labels_ids = eval_preds.label_ids
        pred_ids = eval_preds.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        result = rouge.compute(predictions=pred_str, references=label_str)

        return result

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if args.do_train:
        trainer.train()
        trainer.save_model(args.output_dir)

    elif args.do_eval:
        trainer.evaluate(eval_dataset=dev_dataset)
    
    
if __name__ == "__main__":
    main()
