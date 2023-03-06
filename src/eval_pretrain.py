
import sys
sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")
sys.path.insert(0, "/mnt/private-pa002-vol726121-prd/Code/RLHF")

import os
import argparse
import numpy as np
import torch

from torchmetrics.text.perplexity import Perplexity
from torchmetrics.text.rouge import ROUGEScore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    TextGenerationPipeline
)

from src.utils import logger, RESOURCE_PATH
from src.utils.data import OCNLIDataset
from src.utils.file_utils import set_seed


DATASET = {
    "oncli": OCNLIDataset,
    # "cmnli": CMNLIDataset
}
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
    parser.add_argument("--task", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=1024)
    # parser.add_argument("--max_length_prompt", type=int, default=200)
    # parser.add_argument("--max_length_label", type=int, default=824)
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
    parser.add_argument("--num_return_sequences", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=10.0)

    args = parser.parse_args()
    
    return args


def main():
    args = get_parser()
    logger.info(f"Parameters: {args}")

    set_seed(args.seed)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
    tokenizer.add_special_tokens({
        "unk_token": "<unk>",
        'eos_token': "<eot>",
        'pad_token': "<pad>",
        "sep_token": "<sep>"
    })
    # assert tokenizer.pad_token_id == tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer.sp))
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # model.config.max_length_prompt = args.max_length_prompt
    if args.checkpoint is not None:
        st = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(st)
    logger.info(f"Finished loading model and tokenizer")

    # Set up the datasets
    dataset = DATASET.get(args.task, None)
    if dataset is None:
        raise ValueError(f"Unsupported task: {args.task}")
    if args.do_train:
        train_dataset = dataset(args, os.path.join(args.data_dir, args.train_filename),
                                     tokenizer)
    else:
        train_dataset = None
    if args.do_eval:
        dev_dataset = dataset(args, os.path.join(args.data_dir, args.eval_filename),
                                   tokenizer)
    else:
        dev_dataset = None
    if args.do_pred:
        test_dataset = dataset.load_dataset(args, os.path.join(args.data_dir, args.test_filename),
                                                 tokenizer)
    else:
        test_dataset = None

    # training arguments
    deepspeed_config = os.path.join(RESOURCE_PATH, "config", "pretrain_model", args.deepspeed_config) if args.deepspeed_config is not None else None
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
        fp16=torch.cuda.is_available(),
        adam_beta1=0.9,
        adam_beta2=0.95,
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
        include_inputs_for_metrics=True,
        per_device_eval_batch_size=args.eval_batch_size,
        do_predict=args.do_pred,
        use_legacy_prediction_loop=args.do_pred,
    )
    logger.info(f"Training Arguments: {training_args}")

    # Set up the metric
    if args.task in ["ocnli", "cmnli"]:
        perplexity = Perplexity(ignore_index=tokenizer.pad_token_id)

        def compute_metrics(eval_preds):

            labels = torch.tensor(eval_preds.label_ids, dtype=torch.long).squeeze(1)
            logits = torch.tensor(eval_preds.predictions, dtype=torch.float).squeeze(1)
            probs = torch.softmax(logits, dim=-1)
            ppls = []
            for i in range(len(labels)):
                ppl = perplexity(probs[i:i+1], labels[i:i+1]).detach().tolist()
                ppls.append(ppl)
            # f1, exact match, acc
            # pred_ids = preprocess_logits_for_metrics(logits, None)
            # pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            # label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # result = rouge(pred_str, label_str)

            return {"ppl": ppls}
    elif args.task in []:
        rouge = ROUGEScore(rouge_keys=('rougeL', 'rouge1', 'rouge2'))

    # Prepare the trainer and start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if args.do_train:
        trainer.train()
        trainer.save_model(args.output_dir)

    elif args.do_eval:
        res = trainer.evaluate(eval_dataset=dev_dataset)
        if args.task in ["ocnli", "cmnli"]:
            ct = 0
            ct_acc = 0
            ppls = []
            # cur_label = None
            ls = list(dev_dataset.label_dict.values())
            for i, (data, ppl) in enumerate(zip(dev_dataset, res['eval_ppl'])):
                ppls.append(ppl)
                cur_label = data['label_str']
                if i % len(ls) == len(ls) - 1:
                    lidx = ls.index(cur_label)
                    if np.argmin(ppls) == lidx:
                        ct += 1
                    ct += 1
                    # cur_label = None
                    ppls = []
            with open(os.path.join(args.output_dir, f"{args.task}_eval_result.txt"), "w", encoding="utf-8") as w:
                w.write(f"ppl={ct_acc/ct}\n")
            logger.info(f"ppl={ct_acc/ct}")
        elif args.task in []:
            pass

    if args.do_pred:
        device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
        text_generator = TextGenerationPipeline(model, tokenizer, device=device)
        batch = []
        with open(os.path.join(args.output_dir, args.output_filename), "w", encoding="utf-8") as w:
            w.write("\t".join(["prompt"]+[f"model_answer_{i}" for i in range(args.num_return_sequences)])+"\n")
            for test_data in test_dataset:
                prompt = test_data['prompt'] + tokenizer.sep_token + "模型回答:"
                # label = test_data['label']
                batch.append(prompt)
                if len(batch) >= args.eval_batch_size:
                    results = text_generator(batch, max_length=args.max_length_label, do_sample=True,
                                             num_return_sequences=args.num_return_sequences,
                                             top_k=args.top_k, temperature=args.temperature)
                    for prompt, res in zip(batch, results):
                        w.write("\t".join([prompt]+[each['generated_text'].split("模型回答:")[1].replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "") for each in res])+"\n")
                    batch = []
            if len(batch) > 0:
                results = text_generator(batch, max_length=args.max_length_label, do_sample=True,
                                         num_return_sequences=args.num_return_sequences,
                                         top_k=args.top_k, temperature=args.temperature)
                for prompt, res in zip(batch, results):
                    w.write("\t".join([prompt]+[each['generated_text'].split("模型回答:")[1].replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "") for each in res])+"\n")

        # output = trainer.predict(test_dataset)

    
if __name__ == "__main__":
    main()
