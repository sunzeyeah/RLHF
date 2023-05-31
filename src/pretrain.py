
import sys
sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")
sys.path.insert(0, "/mnt/sfevol775196/sunzeye273/Code/chatgpt")
# sys.path.insert(0, "/mnt/share-pa002-vol682688-prd/sunzeye273/Code/chatgpt")
sys.path.insert(0, "/mnt/pa002-28359-vol543625-private/Code/chatgpt")
import os
import argparse
import torch

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    default_data_collator
)

from src.utils import logger, RESOURCE_PATH
from src.data.data import PretrainDataset
from src.utils.file_utils import set_seed


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
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_length_generation", type=int, default=None)
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
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=int, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, default="steps",
                        help='- `"no"`: No save is done during training.'
                             '- `"epoch"`: Save is done at the end of each epoch.'
                             '- `"steps"`: Save is done every `save_steps`.')
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", action="store_true",
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
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)

    args = parser.parse_args()
    
    return args


def main():
    args = get_parser()
    if args.local_rank <= 0:
        logger.info(f"Parameters: {args}")

    set_seed(args.seed)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)

    # load model
    if "llama" in args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
    elif "pangu" in args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
        model.resize_token_embeddings(tokenizer.vocab_size)
        # model.config.pad_token_id = tokenizer.pad_token_id
        # model.config.bos_token_id = tokenizer.bos_token_id
        # model.config.eos_token_id = tokenizer.eos_token_id
    elif "glm" in args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        if "chatglm" in args.model_name_or_path:
            model = model.half()
    else:
        raise ValueError(f"Unsupported model name: {args.model_name_or_path}")
    # assert model.config.pad_token_id == tokenizer.pad_token_id
    # model.config.lora_rank = args.lora_rank
    # model.config.lora_alpha = args.lora_alpha
    # model.config.lora_train_bias = args.lora_train_bias
    # model = SFTModelWithLoRA(model.config, model)

    if args.checkpoint is not None:
        st = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(st, strict=False)

    logger.info(f"Finished loading model and tokenizer")

    # Set up the datasets
    if args.do_train:
        train_dataset = PretrainDataset(args, os.path.join(args.data_dir, args.train_filename),
                                   tokenizer)
    else:
        train_dataset = None
    if args.do_eval:
        dev_dataset = PretrainDataset(args, os.path.join(args.data_dir, args.eval_filename),
                                 tokenizer)
    else:
        dev_dataset = None
    if args.do_pred:
        test_dataset = PretrainDataset(args, os.path.join(args.data_dir, args.test_filename),
                                  tokenizer)
    else:
        test_dataset = None

    # training arguments
    deepspeed_config = os.path.join(RESOURCE_PATH, "config", "deepspeed", args.deepspeed_config) if args.deepspeed_config is not None else None
    if torch.cuda.is_available():
        bf16 = torch.cuda.get_device_capability()[0] >= 8
        fp16 = not bf16
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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        half_precision_backend="auto",
        fp16=fp16,
        bf16=bf16,
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
        per_device_eval_batch_size=args.eval_batch_size,
        do_predict=args.do_pred,
        use_legacy_prediction_loop=args.do_pred,
    )
    if args.local_rank <= 0:
        logger.info(f"Training Arguments: {training_args}")

    # Set up the metric
    # rouge = evaluate.load("rouge")
    #
    # def compute_metrics(eval_preds):
    #     labels_ids = eval_preds.label_ids
    #     pred_ids = eval_preds.predictions
    #     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    #     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    #     result = rouge.compute(predictions=pred_str, references=label_str)
    #
    #     return result

    # Prepare the trainer and start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        # compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    # model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    if args.do_train:
        trainer.train()
        trainer.save_model(args.output_dir)

    elif args.do_eval:
        res = trainer.evaluate(eval_dataset=dev_dataset)
        logger.info(res)

    if args.do_pred:
        model.eval()
        device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        tokenizer.padding_side = "left"
        with open(os.path.join(args.output_dir, args.output_filename), "w", encoding="utf-8") as w:
            w.write("\t".join(["prompt"]+[f"model_answer_{i}" for i in range(args.num_return_sequences)])+"\n")
            for test_data in tqdm(test_dataset.post_list, desc="Prediction"):
                prompt = test_data['prompt']
                prefix = test_data['prefix']
                # label = dev_data['label']
                if "pangu" in args.model_name_or_path:
                    inputs = tokenizer(prompt, tokenizer.sep_token + prefix, max_length=args.max_length,
                                       truncation="only_first", add_special_tokens=False,
                                       return_tensors="pt", return_token_type_ids=False)
                    # inputs = tokenizer(prompt, add_special_tokens=False, return_token_type_ids=False, return_tensors="pt")
                    inputs = inputs.to(device)
                    outputs = model.generate(**inputs,
                                             max_new_tokens=args.max_length_generation,
                                             pad_token_id=tokenizer.pad_token_id,
                                             do_sample=args.do_sample,
                                             num_return_sequences=args.num_return_sequences,
                                             top_k=args.top_k,
                                             top_p=args.top_p,
                                             temperature=args.temperature)
                elif "chatglm" in args.model_name_or_path:
                    encoded_prompt = tokenizer(prompt)
                    prompt_length = len(encoded_prompt['input_ids'])
                    inputs = tokenizer(prompt,
                                       max_length=min(prompt_length, args.max_length),
                                       truncation="only_first",
                                       return_tensors="pt")
                    # max_gen_length = args.max_length - encoded_dict['input_ids'].shape[1]
                    # inputs = tokenizer.build_inputs_for_generation(encoded_dict,
                    #                                                max_gen_length=max_gen_length, padding=True)
                    inputs = inputs.to(device)
                    outputs = model.generate(**inputs,
                                             max_new_tokens=args.max_length_generation,
                                             eos_token_id=tokenizer.eop_token_id,
                                             pad_token_id=tokenizer.pad_token_id,
                                             do_sample=args.do_sample,
                                             num_return_sequences=args.num_return_sequences,
                                             top_k=args.top_k,
                                             top_p=args.top_p,
                                             temperature=args.temperature)
                elif "glm" in args.model_name_or_path:
                    encoded_prompt = tokenizer(prompt, prefix + tokenizer.mask_token)
                    prompt_length = len(encoded_prompt['input_ids'])
                    encoded_dict = tokenizer(prompt, prefix + tokenizer.mask_token,
                                             max_length=min(prompt_length, args.max_length),
                                             truncation="only_first",
                                             return_tensors="pt",
                                             return_token_type_ids=False)
                    max_gen_length = args.max_length - encoded_dict['input_ids'].shape[1]
                    inputs = tokenizer.build_inputs_for_generation(encoded_dict,
                                                                   max_gen_length=max_gen_length, padding=True)
                    inputs = inputs.to(device)
                    outputs = model.generate(**inputs,
                                             max_new_tokens=min(args.max_length_generation, max_gen_length),
                                             eos_token_id=tokenizer.eop_token_id,
                                             pad_token_id=tokenizer.pad_token_id,
                                             do_sample=args.do_sample,
                                             num_return_sequences=args.num_return_sequences,
                                             top_k=args.top_k,
                                             top_p=args.top_p,
                                             temperature=args.temperature)
                else:
                    raise ValueError(f"Unsupported model name: {args.model_name_or_path}")
                results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                w.write("\t".join([prompt]+[result.split(prefix, maxsplit=1)[1] for result in results])+"\n")

    
if __name__ == "__main__":
    main()
