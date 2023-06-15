
import sys
sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")
sys.path.insert(0, "/mnt/sfevol775196/sunzeye273/Code/chatgpt")
# sys.path.insert(0, "/mnt/share-pa002-vol682688-prd/sunzeye273/Code/chatgpt")
sys.path.insert(0, "/mnt/pa002-28359-vol543625-private/Code/chatgpt")
import os
import argparse
import evaluate
import torch
import loralib as lora
import json
import deepspeed

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader
from transformers.deepspeed import HfDeepSpeedConfig
# from deepspeed.ops.adam import FusedAdam
# from deepspeed.ops.adam import DeepSpeedCPUAdam
from peft import (
    LoraConfig,
    get_peft_model
)


from src.utils import logger, RESOURCE_PATH
from src.data.data import SFTDataset
from src.utils.file_utils import set_seed, print_rank_0, print_trainable_parameters
from src.utils.modeling_utils import rotate_checkpoints, save_zero_three_model
# from src.models import convert_to_lora_recursively


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
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, default="steps",
                        help='- `"no"`: No save is done during training.'
                             '- `"epoch"`: Save is done at the end of each epoch.'
                             '- `"steps"`: Save is done every `save_steps`.')
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
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

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

    print_rank_0(f"Parameters: {args}")

    set_seed(args.seed)

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
        if torch.cuda.is_available():
            bf16 = torch.cuda.get_device_capability()[0] >= 8
            fp16 = False if bf16 else True
        else:
            fp16 = False
            bf16 = False
        ds_config["fp16"]["enabled"] = fp16
        ds_config["bf16"]["enabled"] = bf16
        ds_config["optimizer"]["params"] = {
                "lr": args.learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": args.weight_decay
            }
        ds_config["scheduler"]['type'] = args.lr_scheduler_type
        ds_config["scheduler"]["params"] = {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": args.learning_rate,
                    "warmup_num_steps": args.warmup_steps
                }
        dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
    if "pangu" in args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
        model.resize_token_embeddings(tokenizer.vocab_size)
        # model.config.pad_token_id = tokenizer.pad_token_id
        # model.config.bos_token_id = tokenizer.bos_token_id
        # model.config.eos_token_id = tokenizer.eos_token_id
        target_modules = "q_proj,k_proj,v_proj"
        task_type = "CAUSAL_LM"
    elif "glm" in args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        if "chatglm" in args.model_name_or_path:
            model = model.half()
        target_modules = "query_key_value"
        task_type = "SEQ_2_SEQ_LM"
    else:
        raise ValueError(f"Unsupported model name: {args.model_name_or_path}")

    if args.lora_rank > 0:
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules.split(","),
            lora_dropout=0.05,
            bias=args.lora_train_bias,
            task_type=task_type
        )
        model = get_peft_model(model, config)
        print_trainable_parameters(model)
        # convert_to_lora_recursively(model, args.lora_rank, args.lora_alpha)
        # lora.mark_only_lora_as_trainable(model, args.lora_train_bias)

    if args.checkpoint is not None:
        st = torch.load(args.checkpoint, map_location="cpu")
        res = model.load_state_dict(st, strict=False)

    print_rank_0(f"Finished loading model and tokenizer")

    # Set up the datasets
    if args.do_train:
        train_dataset = SFTDataset(args, os.path.join(args.data_dir, args.train_filename),
                                   tokenizer)
    else:
        train_dataset = None
    if args.do_eval:
        dev_dataset = SFTDataset(args, os.path.join(args.data_dir, args.eval_filename),
                                 tokenizer)
    else:
        dev_dataset = None
    if args.do_pred:
        test_dataset = SFTDataset(args, os.path.join(args.data_dir, args.test_filename),
                                  tokenizer)
    else:
        test_dataset = None

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
        # create data loader
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            # collate_fn=data_collator,
            sampler=train_sampler,
            batch_size=args.train_batch_size)

        # training
        model_engine.train()
        if args.gradient_checkpointing:
            model_engine.module.gradient_checkpointing_enable()
        global_step = 0
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
                if global_step % args.save_steps == 0:
                    rotate_checkpoints(args.save_total_limit, use_mtime=True, output_dir=args.output_dir)
                    # save_zero_three_model(model_engine, args.local_rank,
                    #                       save_dir=os.path.join(args.output_dir, f"checkpoint-{global_step}"),
                    #                       zero_stage=ds_config['zero_optimization']['stage'])
                    # model_engine.save_checkpoint(args.output_dir, global_step)
                    model_engine.save_16bit_model(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                    print_rank_0(f"Finished saving checkpoint @Step-{global_step}")

        print_rank_0(f"Finished training! epochs: {epoch+1}, steps: {global_step}")

        # save_zero_three_model(model_engine, args.local_rank,
        #                       save_dir=os.path.join(args.output_dir, f"checkpoint-{global_step}"),
        #                       zero_stage=ds_config['zero_optimization']['stage'])
        # model_engine.save_checkpoint(args.output_dir, global_step)
        model_engine.save_16bit_model(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
        print_rank_0(f"Finished saving checkpoint @Step-{global_step}")

    elif args.do_eval:
        pass

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
                                       truncation="longest_first", add_special_tokens=False,
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
