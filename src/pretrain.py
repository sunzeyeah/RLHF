
import sys
sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")
sys.path.insert(0, "/mnt/sfevol775196/sunzeye273/Code/chatgpt")
# sys.path.insert(0, "/mnt/share-pa002-vol682688-prd/sunzeye273/Code/chatgpt")
sys.path.insert(0, "/mnt/pa002-28359-vol543625-private/Code/chatgpt")
import os
import argparse
import torch
import evaluate
import json

from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from src.utils import RESOURCE_PATH, load_tokenizer_and_model, load_checkpoint
from src.data.data import PretrainDataset, chatglm2_encode
from src.utils.file_utils import set_seed, print_rank_0
# from src.models.llama import LlamaForCausalLM


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
    parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--multi_card", action="store_true")
    parser.add_argument("--bits", type=int, default=32)
    parser.add_argument("--device_map", type=str, default=None, help="device map to allocate model,"
                                                                     "[None] means cpu"
                                                                     "[0, 1, 2, ...], number means single-card"
                                                                     "[auto, balanced, balanced_low_0] means multi-card")
    parser.add_argument("--max_length", type=int, default=1024)
    # train
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--train_filename", type=str, default=None)
    parser.add_argument("--concat_samples", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
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
    parser.add_argument("--lora_alpha", type=int, default=16)
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
    parser.add_argument("--max_length_generation", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_filename", type=str, default=None)
    parser.add_argument("--output_filename", type=str, default=None)
    parser.add_argument("--data_types", type=str, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()
    
    return args


def main():
    args = get_parser()
    print_rank_0(f"Parameters: {args}")

    set_seed(args.seed)

    # load tokenizer and model
    tokenizer, model, eos_token_id = load_tokenizer_and_model(args)

    if args.checkpoint is not None:
        load_checkpoint(args, model)

    print_rank_0(f"Finished loading model and tokenizer")

    # Set up the datasets
    if args.do_train:
        train_dataset = PretrainDataset(args, os.path.join(args.data_dir, args.train_filename),
                                        tokenizer, concat_samples=args.concat_samples)
    else:
        train_dataset = None
    if args.do_eval:
        dev_dataset = PretrainDataset(args, os.path.join(args.data_dir, args.eval_filename),
                                      tokenizer, concat_samples=False)
    else:
        dev_dataset = None
    # if args.do_pred:
    #     test_dataset = PretrainDataset(args, os.path.join(args.data_dir, args.test_filename),
    #                                    tokenizer)
    # else:
    #     test_dataset = None

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
            adam_beta1=0.9,
            adam_beta2=0.95,
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

        # Set up the metric
        rouge = evaluate.load("rouge")

        def compute_metrics(eval_preds):
            labels_ids = eval_preds.label_ids
            pred_ids = eval_preds.predictions
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
            result = rouge.compute(predictions=pred_str, references=label_str)

            return result

        # Prepare the trainer and start training
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics,
            data_collator=default_data_collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        # model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

        trainer.train()
        trainer.save_model(args.output_dir)

    elif args.do_eval:
        # res = trainer.evaluate(eval_dataset=dev_dataset)
        # logger.info(res)
        pass

    if args.do_pred:
        device = f"cuda:{args.local_rank}" if torch.cuda.is_available() and args.device_map is not None else "cpu"
        model.eval()
        # tokenizer.padding_side = "left"
        if args.test_filename is not None:
            test_file = os.path.join(args.data_dir, args.test_filename)
        else:
            test_file = os.path.join(RESOURCE_PATH, "test_prompts.jsonl")
        data_types = args.data_types.split(",") if args.data_types is not None else None
        with torch.no_grad():
            if args.output_filename is not None:
                w = open(os.path.join(args.output_dir, args.output_filename), "w", encoding="utf-8")
            else:
                w = None
            # for line in tqdm(open(test_file, "r", encoding="utf-8"), desc="Prediction"):
            for line in open(test_file, "r", encoding="utf-8"):
                test_data = json.loads(line.strip("\n"))
                data_type = test_data.get('data_type', None)
                if data_types is not None and data_type not in data_types:
                    continue
                prompt = test_data['prompt']
                prefix = test_data.get('prefix', None)
                system = test_data.get('system', None)
                label = test_data.get('label', None)
                # encoded_prompt = tokenizer(prompt)
                if "chatglm2" in args.model_name_or_path.lower():
                    input_ids, _, prompt_ids = chatglm2_encode(tokenizer, prompt, None, system, args.max_length)
                    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                    outputs = model.generate(input_ids=input_ids,
                                             max_new_tokens=args.max_length_generation,
                                             eos_token_id=eos_token_id,
                                             pad_token_id=tokenizer.pad_token_id,
                                             do_sample=args.do_sample,
                                             num_return_sequences=args.num_return_sequences,
                                             top_k=args.top_k,
                                             top_p=args.top_p,
                                             temperature=args.temperature)
                    prompt_length = len(prompt_ids)
                elif "chatglm" in args.model_name_or_path.lower():
                    inputs = tokenizer(prompt, max_length=args.max_length-args.max_length_generation,
                                       truncation="only_first",
                                       return_tensors="pt")
                    inputs = inputs.to(device)
                    outputs = model.generate(inputs=inputs['input_ids'],
                                             max_new_tokens=args.max_length_generation,
                                             eos_token_id=eos_token_id,
                                             pad_token_id=tokenizer.pad_token_id,
                                             do_sample=args.do_sample,
                                             num_return_sequences=args.num_return_sequences,
                                             top_k=args.top_k,
                                             top_p=args.top_p,
                                             temperature=args.temperature,
                                             repetition_penalty=args.repetition_penalty)
                    prompt_length = len(inputs['input_ids'][0])
                # elif "glm" in args.model_name_or_path.lower():
                #     encoded_prompt = tokenizer(prompt, prefix + tokenizer.mask_token)
                #     prompt_length = len(encoded_prompt['input_ids'])
                #     encoded_dict = tokenizer(prompt, prefix + tokenizer.mask_token,
                #                              max_length=min(prompt_length, args.max_length),
                #                              truncation="only_first",
                #                              return_tensors="pt",
                #                              return_token_type_ids=False)
                #     max_gen_length = args.max_length - encoded_dict['input_ids'].shape[1]
                #     inputs = tokenizer.build_inputs_for_generation(encoded_dict,
                #                                                    max_gen_length=max_gen_length, padding=True)
                #     inputs = inputs.to(device)
                #     outputs = model.generate(**inputs,
                #                              max_new_tokens=min(args.max_length_generation, max_gen_length),
                #                              eos_token_id=tokenizer.eop_token_id,
                #                              pad_token_id=tokenizer.pad_token_id,
                #                              do_sample=args.do_sample,
                #                              num_return_sequences=args.num_return_sequences,
                #                              top_k=args.top_k,
                #                              top_p=args.top_p,
                #                              temperature=args.temperature)
                else:
                    if prefix is not None and len(prefix) > 0:
                        prompt += prefix
                    inputs = tokenizer(prompt, max_length=args.max_length-args.max_length_generation,
                                       truncation="only_first",
                                       return_tensors="pt")
                    inputs = inputs.to(device)
                    outputs = model.generate(inputs=inputs['input_ids'],
                                             max_new_tokens=args.max_length_generation,
                                             eos_token_id=eos_token_id,
                                             do_sample=args.do_sample,
                                             num_return_sequences=args.num_return_sequences,
                                             top_k=args.top_k,
                                             top_p=args.top_p,
                                             temperature=args.temperature,
                                             repetition_penalty=args.repetition_penalty)
                    prompt_length = len(inputs['input_ids'][0])
                results = tokenizer.batch_decode([output[prompt_length:] for output in outputs], skip_special_tokens=True)
                # p = tokenizer.decode(encoded_prompt['input_ids'], skip_special_tokens=True)
                answers = []
                for r in results:
                    # answer = r.replace(p, "").strip()
                    answer = r
                    print_rank_0(f"\nprompt: {prompt}\nanswer: {answer}")
                    answers.append({"answer": answer, "score": None})
                if w is not None:
                    w.write(json.dumps({"prompt": prompt, "prefix": prefix, "system": system, "answers": answers,
                                        "label": label}, ensure_ascii=False)+"\n")

            if w is not None:
                w.close()


if __name__ == "__main__":
    from transformers import pipeline
    main()
