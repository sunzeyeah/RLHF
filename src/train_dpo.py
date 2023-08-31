import sys

sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")
sys.path.insert(0, "/mnt/sfevol775196/sunzeye273/Code/chatgpt")
# sys.path.insert(0, "/mnt/share-pa002-vol682688-prd/sunzeye273/Code/chatgpt")
sys.path.insert(0, "/mnt/pa002-28359-vol543625-private/Code/chatgpt")
import os
import argparse
import evaluate
import torch
import copy

from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import (
    TrainingArguments,
    default_data_collator,
)

from src.utils import RESOURCE_PATH, load_tokenizer_and_model, load_checkpoint
from src.data.data import DPODataset, SFTDataset
from src.utils.file_utils import set_seed, print_rank_0
from src.models.trainer import DPOTrainer


# Create a preprocessing function to extract out the proper logits from the model output
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]

    return logits.argmax(dim=-1)


def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)

    parser.add_argument("--reference_model_name_or_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_length_prompt", type=int, default=512)
    parser.add_argument("--max_length_generation", type=int, default=None)
    parser.add_argument("--bits", type=int, default=32,
                        help="bits used to load model, including: 32, 16, 8, 4")
    # parser.add_argument("--multi_card", action="store_true")
    parser.add_argument("--device_map", type=str, default=None, help="device map to allocate model,"
                                                                     "[None] means cpu"
                                                                     "[0, 1, 2, ...], number means single-card"
                                                                     "[auto, balanced, balanced_low_0] means multi-card")
    # train
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--train_filename", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="the beta parameter for DPO loss")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="transformers.trainer_utils.SchedulerType, including:"
                             "linear, cosine, cosine_with_restarts, polynomial, constant,"
                             "constant_with_warmup")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=int, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, default="steps",
                        help='- `"no"`: No save is done during training.'
                             '- `"epoch"`: Save is done at the end of each epoch.'
                             '- `"steps"`: Save is done every `save_steps`.')
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--metric_for_best_model", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
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
    parser.add_argument("--evaluation_strategy", type=str, default="steps",
                        help='- `"no"`: No evaluation is done during training.'
                             '- `"steps"`: Evaluation is done (and logged) every `eval_steps`.'
                             '- `"epoch"`: Evaluation is done at the end of each epoch.')
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)
    # pred
    parser.add_argument("--do_pred", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_filename", type=str, default=None)
    parser.add_argument("--output_filename", type=str, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)

    args = parser.parse_args()
    
    return args


def main():
    args = get_parser()
    print_rank_0(f"Parameters: {args}")

    set_seed(args.seed)

    # load tokenizer and model
    tokenizer, model, eos_token_id = load_tokenizer_and_model(args)

    if args.checkpoint is not None:
        load_checkpoint(args, model, strict=False)

    print_rank_0(f"Finished loading model and tokenizer")

    # Set up the datasets
    if args.do_train:
        train_dataset = DPODataset(args, os.path.join(args.data_dir, args.train_filename),
                                   tokenizer)
    else:
        train_dataset = None
    if args.do_eval:
        dev_dataset = DPODataset(args, os.path.join(args.data_dir, args.eval_filename),
                                 tokenizer)
    else:
        dev_dataset = None

    if args.do_train:
        if torch.cuda.is_available():
            bf16 = torch.cuda.get_device_capability()[0] >= 8
            fp16 = not bf16
        else:
            fp16 = False
            bf16 = False
        # training arguments
        deepspeed_config = os.path.join(RESOURCE_PATH, "config", "deepspeed", args.deepspeed_config) if args.deepspeed_config is not None else None
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
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            half_precision_backend="auto",
            fp16=fp16,
            bf16=bf16,
            optim="paged_adamw_8bit",
            # adam_beta1=0.9,
            # adam_beta2=0.95,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=True,
            logging_steps=args.logging_steps,
            report_to=["tensorboard"],
            deepspeed=deepspeed_config,
            gradient_checkpointing=args.gradient_checkpointing,
            do_eval=args.do_eval,
            evaluation_strategy=args.evaluation_strategy,
            eval_steps=args.eval_steps,
            eval_accumulation_steps=args.eval_accumulation_steps,
            per_device_eval_batch_size=args.eval_batch_size,
            # do_predict=args.do_pred,
            # use_legacy_prediction_loop=args.do_pred,
        )
        print_rank_0(f"Training Arguments: {training_args}")

        # load reference model or precomputed reference result
        if args.output_filename is not None:
            logps = torch.load(os.path.join(args.output_dir, args.output_filename))
            ref_model = None
        else:
            logps = None
            ref_args = copy.deepcopy(args)
            ref_args.device_map = "auto"
            if args.reference_model_name_or_path is not None:
                ref_args.model_name_or_path = args.reference_model_name_or_path
            else:
                ref_args.bits = 4
            _, ref_model, _ = load_tokenizer_and_model(ref_args)
            ref_model.eval()

        # Prepare the trainer and start training
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            logps=logps,
            args=training_args,
            beta=args.beta,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            # compute_metrics=compute_metrics,
            # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            label_pad_token_id=tokenizer.pad_token_id
        )
        # model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()
        trainer.save_model(args.output_dir)

    elif args.do_eval:
        # res = trainer.evaluate(eval_dataset=dev_dataset)
        # print_rank_0(res)
        pass

    if args.do_pred:
        def _get_batch_logps(
                logits: torch.FloatTensor,
                labels: torch.LongTensor,
                average_log_prob: bool = False,
        ) -> torch.FloatTensor:
            """Compute the log probabilities of the given labels under the given logits.

            Args:
                logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
                labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
                average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

            Returns:
                A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
            """
            if logits.shape[:-1] != labels.shape:
                raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
            loss_mask = labels != tokenizer.pad_token_id

            # dummy token; we'll ignore the losses on these tokens later
            labels[labels == tokenizer.pad_token_id] = 0

            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

            if average_log_prob:
                return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
            else:
                return (per_token_logps * loss_mask).sum(-1)

        model.eval()
        device = f"cuda:{args.local_rank}" if torch.cuda.is_available() and args.device_map is not None else "cpu"

        logps = dict()
        for test_filename in args.test_filename.split(","):
            if "train" in test_filename:
                mode = "train"
            else:
                mode = "eval"
            logps[mode] = dict()
            test_filename = os.path.join(args.data_dir, test_filename)
            test_dataset = DPODataset(args, test_filename, tokenizer)
            sampler = SequentialSampler(test_dataset)
            test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, sampler=sampler)
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Prediction on {mode}"):
                    indices = batch['index'].tolist()
                    chosen_input_ids = batch['chosen_input_ids'].to(device)
                    chosen_attention_mask = batch['chosen_attention_mask'].to(device) if 'chosen_attention_mask' in batch else None
                    rejected_input_ids = batch['rejected_input_ids'].to(device)
                    rejected_attention_mask = batch['rejected_attention_mask'].to(device) if 'rejected_attention_mask' in batch else None
                    chosen_logits = model(chosen_input_ids, chosen_attention_mask).logits.detach().cpu().to(torch.float32)
                    chosen_logps = _get_batch_logps(chosen_logits, batch["chosen_labels"], average_log_prob=False)
                    rejected_logits = model(rejected_input_ids, rejected_attention_mask).logits.detach().cpu().to(torch.float32)
                    rejected_logps = _get_batch_logps(rejected_logits, batch["rejected_labels"], average_log_prob=False)
                    for index, chosen_logop, rejected_logp in zip(indices, chosen_logps, rejected_logps):
                        logps[mode][index] = {"chosen_logop": chosen_logop, "rejected_logp": rejected_logp}

        torch.save(logps, os.path.join(args.output_dir, args.output_filename))


if __name__ == "__main__":
    main()
