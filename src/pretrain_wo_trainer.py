
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
import numpy as np
import deepspeed

from datetime import datetime
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LlamaTokenizer,
    BitsAndBytesConfig
)
from torch.utils.data import RandomSampler, SequentialSampler, DistributedSampler, DataLoader
from transformers.deepspeed import HfDeepSpeedConfig
# from deepspeed.ops.adam import FusedAdam
# from deepspeed.ops.adam import DeepSpeedCPUAdam

from src.utils import logger, RESOURCE_PATH, load_tokenizer_and_model, load_checkpoint
from src.data.data import PretrainDataset
from src.utils.file_utils import set_seed, print_gpu_utilization, print_rank_0
from src.utils.modeling_utils import rotate_checkpoints, save_zero_three_model
# from src.models import convert_to_lora_recursively
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
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_length_generation", type=int, default=None)
    # parser.add_argument("--multi_card", action="store_true")
    parser.add_argument("--bits", type=int, default=32)
    parser.add_argument("--device_map", type=str, default=None, help="device map to allocate model,"
                                                                     "[None] means cpu"
                                                                     "[0, 1, 2, ...], number means single-card"
                                                                     "[auto, balanced, balanced_low_0] means multi-card")
    # train
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--train_filename", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="OneCycle",
                        help="deepspeed scheduler types, including:"
                             "LRRangeTest, OneCycle, WarmupLR, WarmupDecayLR")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, default="steps",
                        help='- `"no"`: No save is done during training.'
                             '- `"epoch"`: Save is done at the end of each epoch.'
                             '- `"steps"`: Save is done every `save_steps`.')
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--metric_for_best_model", type=str, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
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
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_filename", type=str, default=None)
    parser.add_argument("--output_filename", type=str, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()
    
    return args


def pred_single_sample(prompt, prefix, model, tokenizer, args, device, eos_token_id):
    max_prompt_length = args.max_length - args.max_length_generation
    if "chatglm" in args.model_name_or_path.lower():
        encoded_prompt = tokenizer(prompt)
        prompt_length = len(encoded_prompt['input_ids'])
        inputs = tokenizer(prompt,
                           max_length=min(prompt_length, args.max_length),
                           truncation="only_first",
                           return_tensors="pt")
        # max_gen_length = args.max_length - encoded_dict['input_ids'].shape[1]
        # inputs = tokenizer.build_inputs_for_generation(encoded_dict,
        #                                                max_gen_length=max_gen_length, padding=True)
        input_ids = inputs['input_ids']
        inputs = inputs.to(device)
        outputs = model.generate(inputs=inputs['input_ids'],
                                 max_new_tokens=args.max_length_generation,
                                 eos_token_id=eos_token_id,
                                 pad_token_id=tokenizer.pad_token_id,
                                 do_sample=args.do_sample,
                                 num_return_sequences=args.num_return_sequences,
                                 top_k=args.top_k,
                                 top_p=args.top_p,
                                 temperature=args.temperature)
    # elif "glm" in args.model_name_or_path.lower():
    #     encoded_prompt = tokenizer(prompt, prefix + tokenizer.mask_token)
    #     prompt_length = len(encoded_prompt['input_ids'])
    #     encoded_dict = tokenizer(prompt, prefix + tokenizer.mask_token,
    #                              max_length=min(prompt_length, args.max_length),
    #                              truncation="only_first",
    #                              return_tensors="pt",
    #                              return_token_type_ids=False)
    #     input_ids = encoded_dict['input_ids']
    #     max_gen_length = args.max_length - encoded_dict['input_ids'].shape[1]
    #     inputs = tokenizer.build_inputs_for_generation(encoded_dict,
    #                                                    max_gen_length=max_gen_length, padding=True)
    #     inputs = inputs.to(device)
    #     outputs = model.generate(inputs=inputs['input_ids'],
    #                              max_new_tokens=min(args.max_length_generation, max_gen_length),
    #                              eos_token_id=tokenizer.eop_token_id,
    #                              pad_token_id=tokenizer.pad_token_id,
    #                              do_sample=args.do_sample,
    #                              num_return_sequences=args.num_return_sequences,
    #                              top_k=args.top_k,
    #                              top_p=args.top_p,
    #                              temperature=args.temperature)
    else:
        inputs = tokenizer(prompt, max_length=max_prompt_length, truncation="longest_first", return_tensors="pt")
        input_ids = inputs['input_ids']
        inputs = inputs.to(device)
        outputs = model.generate(inputs=inputs['input_ids'],
                                 max_new_tokens=args.max_length_generation,
                                 do_sample=args.do_sample,
                                 num_return_sequences=args.num_return_sequences,
                                 top_k=args.top_k,
                                 top_p=args.top_p,
                                 temperature=args.temperature,
                                 repetition_penalty=args.repetition_penalty)

    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    p = tokenizer.decode(input_ids, skip_special_tokens=True)
    results = [result.replace(p, "").strip() for result in results]
    answers = []
    for r in results:
        print_rank_0(f"\nprompt: {p}\nanswer: {r}")
        answers.append({"answer": r, "score": None})
    d = {"prompt": prompt, "prefix": prefix, "answers": answers}

    return d


def pred(args, model, tokenizer, device, eos_token_id, step=-1):
    print_rank_0(f"Prediction Result@{step}")
    with torch.no_grad():
        with open(os.path.join(args.output_dir, args.output_filename.format(step=step)), "w", encoding="utf-8") as w:
            with open(os.path.join(args.data_dir, args.test_filename), "r", encoding="utf-8") as r:
                while True:
                    line = r.readline()
                    if not line:
                        break
                    item = json.loads(line.strip("\n"))
                    prompt = item['context']
                    result = pred_single_sample(prompt, "", model, tokenizer, args, device, eos_token_id)
                    if args.local_rank <= 0:
                        w.write(json.dumps(result, ensure_ascii=False)+"\n")


def main():
    args = get_parser()
    print_rank_0(f"Parameters: {args}")

    set_seed(args.seed)

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    # load quantization config
    if torch.cuda.is_available():
        bf16 = torch.cuda.get_device_capability()[0] >= 8
        fp16 = not bf16
    else:
        fp16 = False
        bf16 = False

    # create HfDeepSpeedConfig [must be called before instantiating model]
    if args.deepspeed_config is not None:
        ds_config_filename = os.path.join(RESOURCE_PATH, "config", "deepspeed", args.deepspeed_config)
        ds_config = json.load(open(ds_config_filename, "r", encoding="utf-8"))
        # ds_config["steps_per_print"] = args.logging_steps
        ds_config["train_micro_batch_size_per_gpu"] = args.train_batch_size
        ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        ds_config["gradient_clipping"] = args.max_grad_norm
        # TODO: before calling dist init, world size is always 1, therefore ds_config['train_batch_size'] cannot multiply world size
        ds_config['train_batch_size'] = args.train_batch_size * args.gradient_accumulation_steps #* torch.cuda.device_count()
        # TODO: assuming hidden_size=4096
        ds_config["zero_optimization"]["reduce_bucket_size"] = 4096 * 4096
        ds_config["zero_optimization"]["stage3_prefetch_bucket_size"] = 0.9 * 4096 * 4096
        ds_config["zero_optimization"]["stage3_param_persistence_threshold"] = 10 * 4096
        ds_config["fp16"]["enabled"] = fp16
        ds_config["bf16"]["enabled"] = bf16
        ds_config["optimizer"]["params"] = {
                "lr": args.learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": args.weight_decay
            }
        assert ds_config["scheduler"]['type'] == args.lr_scheduler_type
        ds_config["scheduler"]["params"] = {
                    "cycle_min_lr": 0,
                    "cycle_max_lr": args.learning_rate,
                    "cycle_first_step_size": args.warmup_steps
                }
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        ds_config['tensorboard']['job_name'] = f"deepspeed-{current_time}"
        dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

    # load tokenizer and model
    tokenizer, model, eos_token_id = load_tokenizer_and_model(args, with_trainer=False)
    print_gpu_utilization("after from_pretrained()", args.local_rank)

    if args.checkpoint is not None:
        load_checkpoint(args, model)

    print_rank_0(f"Finished loading model and tokenizer")

    # Set up the datasets
    if args.do_train:
        train_dataset = PretrainDataset(args, os.path.join(args.data_dir, args.train_filename),
                                        tokenizer)
    else:
        train_dataset = None
    if args.do_eval:
        eval_dataset = PretrainDataset(args, os.path.join(args.data_dir, args.eval_filename),
                                      tokenizer)
        # Set up the metric
        rouge = evaluate.load("rouge")

        def compute_metrics(pred_ids, label_ids):
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            result = rouge.compute(predictions=pred_str, references=label_str)

            return result
    else:
        eval_dataset = None

    if args.do_train:
        # # Optimizer
        # AdamOptimizer = DeepSpeedCPUAdam if "3" in args.deepspeed_config else FusedAdam
        # optim_params = get_optimizer_grouped_parameters(
        #     actor_model, self.args.actor_weight_decay)
        # optim = AdamOptimizer(optim_params,
        #                       lr=self.args.actor_learning_rate,
        #                       betas=(0.9, 0.95))
        #
        # # LR Scheduler
        # lr_scheduler = get_scheduler(
        #     name=self.args.lr_scheduler_type,
        #     optimizer=optim,
        #     num_warmup_steps=self.args.warmup_steps,
        #     num_training_steps=self.num_total_iters,
        # )

        # deepspeed initialize
        ds_config['train_batch_size'] = args.train_batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()
        model_engine, *_ = deepspeed.initialize(model=model,
                                                # optimizer=optim,
                                                # lr_scheduler=lr_scheduler,
                                                config=ds_config)
        print_gpu_utilization("after deepspeed.initialize()", args.local_rank)

        # create data loader
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            # collate_fn=data_collator,
            sampler=train_sampler,
            batch_size=args.train_batch_size)

        if args.do_eval:
            eval_sampler = DistributedSampler(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset,
                # collate_fn=data_collator,
                sampler=eval_sampler,
                batch_size=args.eval_batch_size)

            def eval(step):
                model_engine.eval()
                eval_results = dict()
                with torch.no_grad():
                    for eval_batch in eval_dataloader:
                        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                        eval_output = model_engine(**eval_batch)
                        pred_ids = preprocess_logits_for_metrics(eval_output.logits, None)
                        result_rouge = compute_metrics(pred_ids, eval_batch['labels'])
                        for k, v in result_rouge.items():
                            key = f"eval_{k}"
                            if key not in eval_results:
                                eval_results[key] = []
                            eval_results[key].append(v)
                        if "eval_loss" not in eval_results:
                            eval_results['eval_loss'] = []
                        eval_results['eval_loss'].append(eval_output.loss.tolist())
                if args.do_pred:
                    pred(args, model_engine, tokenizer, device, eos_token_id, step)
                model_engine.train()
                for k, v in eval_results.items():
                    eval_results[k] = np.mean(eval_results[k])
                return eval_results

        # training
        model_engine.train()
        if args.gradient_checkpointing:
            model_engine.module.gradient_checkpointing_enable()
        print_gpu_utilization("before training begin", args.local_rank)
        global_step = 0
        best_metric = None
        best_model_checkpoint = None
        if args.do_eval:
            assert args.eval_steps <= args.save_steps and args.save_steps % args.eval_steps == 0, \
                f"save steps should be greater than eval steps and be a multiple of eval steps"
            eval_results = eval(global_step)
            print_rank_0(f"Epoch-0, Gloal step-{global_step}, Evaluation result: {eval_results}")
            if args.metric_for_best_model is not None:
                assert args.metric_for_best_model in eval_results, \
                    f"{args.metric_for_best_model} is not a valid metric, " \
                    f"please choose from the following metrics: {eval_results.keys()}"
        for epoch in range(args.num_epochs):
            print_rank_0(f"Beginning of Epoch {epoch+1}/{args.num_epochs}")
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                logger.debug(f"batch keys: {batch.keys()}")
                output = model_engine(**batch)
                model_engine.backward(output.loss)
                model_engine.step()
                global_step += 1
                if global_step % args.logging_steps == 0:
                    print_rank_0(f"Epoch-{epoch+1}, Gloal step-{global_step}, loss: {output.loss}")
                if args.do_eval and global_step % args.eval_steps == 0:
                    eval_results = eval(global_step)
                    print_rank_0(f"Epoch-{epoch+1}, Gloal step-{global_step}, Evaluation result: {eval_results}")
                if global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if args.do_eval and args.metric_for_best_model is not None:
                        if (
                                best_metric is None or
                                best_model_checkpoint is None or
                                eval_results[args.metric_for_best_model] > best_metric
                        ):
                            best_metric = eval_results[args.metric_for_best_model]
                            best_model_checkpoint = output_dir
                    rotate_checkpoints(args.save_total_limit, use_mtime=True, output_dir=args.output_dir,
                                       best_model_checkpoint=best_model_checkpoint)
                    # save_zero_three_model(model_engine, args.local_rank,
                    #                       save_dir=output_dir,
                    #                       zero_stage=ds_config['zero_optimization']['stage'])
                    # model_engine.save_16bit_model(output_dir)
                    model_engine.save_checkpoint(args.output_dir, f"checkpoint-{global_step}")
                    print_rank_0(f"Finished saving checkpoint @Step-{global_step}")

        print_rank_0(f"Finished training! epochs: {epoch+1}, steps: {global_step}")
        output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        # save_zero_three_model(model_engine, args.local_rank,
        #                       save_dir=output_dir,
        #                       zero_stage=ds_config['zero_optimization']['stage'])
        # model_engine.save_16bit_model(output_dir)
        model_engine.save_checkpoint(args.output_dir, f"checkpoint-{global_step}")
        print_rank_0(f"Finished saving checkpoint @Step-{global_step}")

    elif args.do_eval:
        pass

    if args.do_pred:
        model.eval()
        device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        # tokenizer.padding_side = "left"
        pred(args, model, tokenizer, device, eos_token_id)

    
if __name__ == "__main__":
    main()
