import collections
import sys
sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")
sys.path.insert(0, "/mnt/private-pa002-vol726121-prd/Code/RLHF")

import os
import argparse
import json
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torchmetrics.text.perplexity import Perplexity
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TextGenerationPipeline
)

from src.utils import logger
from src.utils.data import (
    OCNLIDataset,
    CMNLIDataset,
    CHIDDataset,
    CMRCDataset,
    CLUEWSCDataset,
    C3Dataset,
    AFQMCDataset,
    CSLDataset,
    IFLYTEKDataset,
    TNEWSDataset
)
from src.utils.file_utils import set_seed


DATASET = {
    # NLI
    "ocnli": OCNLIDataset,
    "cmnli": CMNLIDataset,
    # Cloze and completion
    "chid": CHIDDataset,
    # MRC
    "cmrc2018": CMRCDataset,
    # Winograd
    "cluewsc2020": CLUEWSCDataset,
    # common sense reasoning
    "c3": C3Dataset,
    # Text Classification
    "tnews": TNEWSDataset,
    "iflytek": IFLYTEKDataset,
    "afqmc": AFQMCDataset,
    "csl": CSLDataset
}


def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_length_generation", type=int, default=100, help="Maximum number of newly generated tokens")

    # eval
    parser.add_argument("--eval_filename", type=str, default=None)
    parser.add_argument("--train_filename", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--max_few_shot", type=int, default=15, help="Maximum number of examples in few-shot evaulation")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.8)

    args = parser.parse_args()
    
    return args


def main():
    args = get_parser()
    logger.info(f"Parameters: {args}")

    set_seed(args.seed)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
    if "pangu" in args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    logger.info(f"Finished loading model and tokenizer")

    # Set up the datasets
    dataset = DATASET.get(args.task, None)
    if dataset is None:
        raise ValueError(f"Unsupported task: {args.task}")
    train_filename = os.path.join(args.data_dir, args.train_filename) if args.train_filename is not None else None
    dev_dataset = dataset(args, os.path.join(args.data_dir, args.eval_filename),
                          tokenizer, train_filename)

    # Set up the metric
    perplexity = Perplexity(ignore_index=tokenizer.pad_token_id)

    def preprocess_logits_for_metrics(logits, labels):
        labels = labels.detach().cpu()
        probs = torch.softmax(logits, dim=-1).detach().cpu().to(torch.float32)
        ppls = []
        for i in range(probs.shape[0]):
            ppl = perplexity(probs[i:i+1], labels[i:i+1])
            ppls.append(ppl)

        return torch.stack(ppls)

    def calculate_f1(pred_text, label_text):
        pred_tokens = tokenizer(pred_text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False, return_tensors="pt")['input_ids'][0].tolist()
        label_tokens = tokenizer(label_text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False, return_tensors="pt")['input_ids'][0].tolist()
        common = collections.Counter(pred_tokens) & collections.Counter(label_tokens)
        num_same = sum(common.values())
        if len(pred_tokens) == 0 or len(label_tokens) == 0:
            return int(pred_tokens == label_tokens)
        if num_same == 0:
            return 0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(label_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    if args.train_filename is None:
        output_filename = os.path.join(args.output_dir, f"{args.task}_zeroshot_eval_result.jsonl")
    else:
        output_filename = os.path.join(args.output_dir, f"{args.task}_fewshot_eval_result.jsonl")

    if args.task in ["cmrc2018"]:
        # text_generator = TextGenerationPipeline(model, tokenizer, device=device)
        ems = []
        f1s = []
        with open(output_filename, "w", encoding="utf-8") as w:
            with torch.no_grad():
                for dev_data in tqdm(dev_dataset.post_list, desc="Generation"):
                    prompt = dev_data['prompt']
                    label = dev_data['label']
                    if "glm" in args.model_name_or_path:
                        prompt += tokenizer.mask_token
                        inputs = tokenizer(prompt, return_tensors="pt")
                        inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=args.max_length + args.max_length_generation)
                        inputs = inputs.to(device)
                        outputs = model.generate(**inputs,
                                                 max_new_tokens=args.max_length_generation,
                                                 eos_token_id=tokenizer.eop_token_id,
                                                 pad_token_id=tokenizer.pad_token_id,
                                                 do_sample=False,
                                                 num_return_sequences=args.num_return_sequences,
                                                 top_p=args.top_p,
                                                 temperature=args.temperature)
                    else:
                        inputs = tokenizer(prompt, add_special_tokens=False, return_token_type_ids=False, return_tensors="pt")
                        inputs = inputs.to(device)
                        outputs = model.generate(**inputs,
                                                 max_new_tokens=args.max_length_generation,
                                                 pad_token_id=tokenizer.pad_token_id,
                                                 do_sample=False,
                                                 num_return_sequences=args.num_return_sequences,
                                                 top_p=args.top_p,
                                                 temperature=args.temperature)
                        # outputs = text_generator(prompt, max_length=args.max_length_generation,
                        #                          do_sample=True, num_return_sequences=args.num_return_sequences,
                        #                          top_p=args.top_p, temperature=args.temperature)
                        # results = [output['generated_text'].split("答:", maxsplit=1)[1].replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "") for output in outputs]
                    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    results = [result.split("答:", maxsplit=1)[1] for result in results]

                    # metrics calculation
                    em_max = -1
                    f1_max = -1
                    for l in label:
                        for pred_text in results:
                            label_text = l['text']
                            em = 1 if pred_text == label_text else 0
                            f1 = calculate_f1(pred_text, label_text)
                            w.write(json.dumps({"prompt": prompt, "label": label_text,
                                                "pred": pred_text, "em": em, "f1": f1}, ensure_ascii=False)+"\n")
                            if em > em_max:
                                em_max = em
                            if f1 > f1_max:
                                f1_max = f1
                    ems.append(em_max)
                    f1s.append(f1_max)

        logger.info(f"em={np.mean(ems)}, f1={np.mean(f1s)}")
    else:
        sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler=sampler, batch_size=args.eval_batch_size)

        ppl_list = []
        input_ids_list = []
        label_list = []
        ls_list = []

        with torch.no_grad():
            for batch in tqdm(dev_dataloader, desc="Evaluation"):
                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].squeeze(1).to(device)
                labels = batch['labels'].squeeze(1).to(device)
                out = model(input_ids, attention_mask=attention_mask)
                ppls = preprocess_logits_for_metrics(out.logits, labels)
                input_ids_list.extend(batch['input_ids'].detach().cpu().tolist())
                ppl_list.extend(ppls.detach().cpu().tolist())
                label_list.extend(batch['label_str'])
                if args.task in ['chid', 'c3', 'iflytek', 'tnews']:
                    ls = np.array(batch['candidates']).transpose().tolist()
                    ls_list.extend(ls)
                else:
                    vals = list(dev_dataset.label_dict.values())
                    ls_list.extend([vals]*input_ids.shape[0])

        ct = 0
        ct_acc = 0
        ppls = []
        with open(output_filename, "w", encoding="utf-8") as w:
            for i, (input_ids, label, ls, ppl) in enumerate(zip(input_ids_list, label_list, ls_list, ppl_list)):
                ppls.append(ppl)
                prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
                if i % len(ls) == len(ls) - 1:
                    lidx = ls.index(label)
                    if np.argmin(ppls) == lidx:
                        ct_acc += 1
                    ct += 1
                    # cur_label = None
                    ppls = []
                w.write(json.dumps({"prompt": prompt, "pred": float(ppl), "label": label}, ensure_ascii=False) + "\n")

        logger.info(f"ppl={ct_acc/ct}")


def test():
    args = get_parser()
    logger.info(f"Parameters: {args}")

    set_seed(args.seed)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
    if "pangu" in args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    text_generator = TextGenerationPipeline(model, tokenizer)

    prompt = "阅读文章：《战国无双3》（）是由光荣和ω-force开发的战国无双系列的正统第三续作。本作以三大故事为主轴，分别是以武田信玄等人为主的《关东三国志》，织田信长等人为主的《战国三杰》，石田三成等人为主的《关原的年轻武者》，丰富游戏内的剧情。此部份专门介绍角色，欲知武器情报、奥义字或擅长攻击类型等，请至战国无双系列1.由于乡里大辅先生因故去世，不得不寻找其他声优接手。从猛将传 and Z开始。2.战国无双 编年史的原创男女主角亦有专属声优。此模式是任天堂>游戏谜之村雨城改编的新增模式。本作中共有20张战场地图（不含村雨城），后来发行的猛将传再新增3张战场地图。但游戏内>战役数量繁多，部分地图会有兼用的状况，战役虚实则是以光荣发行的2本「战国无双3 人物真书」内容为主，以下是相关介绍>。（注：前方加☆者为猛将传新增关卡及地图。）合并本篇和猛将传的内容，村雨城模式剔除，战国史模式可直接游玩。主打两>大模式「战史演武」&「争霸演武」。系列作品外传作品\n问：《战国无双3》是由哪两个公司合作开发的？\n答："

    # if "glm" in args.model_name_or_path:
    inputs = tokenizer(prompt, add_special_tokens=False, return_token_type_ids=False, return_tensors="pt")
    # inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=args.max_length_generation)
    # inputs = inputs.to(device)
    outputs = model.generate(**inputs,
                             max_new_tokens=args.max_length_generation,
                             bos_token_id=tokenizer.bos_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.pad_token_id,
                             do_sample=False,
                             num_return_sequences=args.num_return_sequences,
                             top_p=args.top_p,
                             temperature=args.temperature)
    results = [tokenizer.decode(output, skip_special_tokens=True).rsplit("答:", maxsplit=1)[1].strip() for output in outputs]
    # else:
    outputs = text_generator(prompt, max_new_tokens=args.max_length_generation,
                             bos_token_id=tokenizer.bos_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.pad_token_id,
                             do_sample=True, num_return_sequences=args.num_return_sequences,
                             top_p=args.top_p, temperature=args.temperature)
    results = [output['generated_text'].rsplit("答：", maxsplit=1)[1].replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "") for output in outputs]


if __name__ == "__main__":
    main()
    # test()
