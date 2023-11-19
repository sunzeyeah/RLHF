
import sys
sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")
sys.path.insert(0, "/mnt/sfevol775196/sunzeye273/Code/chatgpt")
# sys.path.insert(0, "/mnt/share-pa002-vol682688-prd/sunzeye273/Code/chatgpt")
sys.path.insert(0, "/mnt/pa002-28359-vol543625-private/Code/chatgpt")
import os
import argparse
import json
import numpy as np
import torch
import collections

from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torchmetrics.text.perplexity import Perplexity
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList

from src.data.data import (
    OCNLIDataset,
    CMNLIDataset,
    CHIDDataset,
    CMRCDataset,
    CLUEWSCDataset,
    C3Dataset,
    AFQMCDataset,
    CSLDataset,
    IFLYTEKDataset,
    TNEWSDataset,
    CEvalDataset,
    MMLUDataset,
)
from src.utils import RESOURCE_PATH, load_tokenizer_and_model, load_checkpoint
from src.utils.file_utils import set_seed, print_rank_0


DATASET = {
    "ceval": CEvalDataset,
    "mmlu": MMLUDataset,
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
    # parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--multi_card", action="store_true")
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--device_map", type=str, default=None, help="device map to allocate model,"
                                                                     "[None] means cpu"
                                                                     "[0, 1, 2, ...] numbers mean single-card"
                                                                     "[auto, balanced, balanced_low_0] means multi-card")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_length_generation", type=int, default=1, help="Maximum number of newly generated tokens")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--low_cpu_mem_usage", action="store_true", help="whether to enable low cpu memory usage"
                                                                         "when loading model")

    # eval
    parser.add_argument("--eval_filename", type=str, default=None)
    parser.add_argument("--train_filename", type=str, default=None)
    parser.add_argument("--submission_filename", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--max_few_shot", type=int, default=15, help="Maximum number of examples in few-shot evaulation")
    parser.add_argument("--cot", action="store_true", help="Whether to use Chain of Thought in evaluation")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()
    
    return args


def extract_cot_answer(line, response):
    #TODO: to be implemented
    pass


def main():
    args = get_parser()
    print_rank_0(f"Parameters: {args}")

    set_seed(args.seed)

    # load model and tokenizer
    tokenizer, model, eos_token_id = load_tokenizer_and_model(args)

    if args.checkpoint is not None:
        suffix = args.checkpoint.split(os.sep)[-2] + "_"
        load_checkpoint(args, model)
    else:
        suffix = ""

    print_rank_0(f"Finished loading model and tokenizer")

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

    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    model.eval()

    if args.train_filename is None:
        output_filename = os.path.join(args.output_dir, f"{args.task}_{args.eval_filename}_zero-shot_{args.max_length}_{suffix}eval_result.jsonl")
    else:
        assert args.max_few_shot > 0
        output_filename = os.path.join(args.output_dir, f"{args.task}_{args.eval_filename}_{args.max_few_shot}-shot_{args.max_length}_{suffix}eval_result.jsonl")

    if args.task in ["cmrc2018"]:
        # text_generator = TextGenerationPipeline(model, tokenizer, device=device)
        ems = []
        f1s = []
        with open(output_filename, "w", encoding="utf-8") as w:
            with torch.no_grad():
                for dev_data in tqdm(dev_dataset.post_list, desc="Generation"):
                    prompt = dev_data['prompt']
                    label = dev_data['label']
                    if "glm" in args.model_name_or_path.lower():
                        prompt += tokenizer.mask_token
                        inputs = tokenizer(prompt, return_tensors="pt")
                        inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=args.max_length + args.max_length_generation)
                        inputs = inputs.to(device)
                        outputs = model.generate(**inputs,
                                                 max_new_tokens=args.max_length_generation,
                                                 eos_token_id=eos_token_id,
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

        print_rank_0(f"em={np.mean(ems)}, f1={np.mean(f1s)}")
    elif args.task in ["ceval"]:
        results = dict()
        with torch.no_grad():
            for dev_data in tqdm(dev_dataset, desc="C-Eval Evaluation"):
                subject_name_key = dev_data['subject_name_key']
                if subject_name_key not in results:
                    results[subject_name_key] = list()
                if "chatglm" in args.model_name_or_path.lower():
                    logits_processor = LogitsProcessorList()
                    if "chatglm2" in args.model_name_or_path.lower():
                        class InvalidScoreLogitsProcessor(LogitsProcessor):
                            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                                if torch.isnan(scores).any() or torch.isinf(scores).any():
                                    scores.zero_()
                                    scores[..., 5] = 5e4
                                return scores
                    else:
                        class InvalidScoreLogitsProcessor(LogitsProcessor):
                            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                                if torch.isnan(scores).any() or torch.isinf(scores).any():
                                    scores.zero_()
                                    scores[..., 20005] = 5e4
                                return scores
                    logits_processor.append(InvalidScoreLogitsProcessor())
                    input_ids = dev_data['input_ids'].to(device)
                    outputs = model.generate(input_ids=input_ids,
                                             max_new_tokens=args.max_length_generation,
                                             do_sample=args.do_sample,
                                             num_return_sequences=args.num_return_sequences,
                                             top_p=args.top_p,
                                             temperature=args.temperature,
                                             repetition_penalty=args.repetition_penalty,
                                             logits_processor=logits_processor,
                                             output_scores=not args.cot,
                                             return_dict_in_generate=not args.cot)
                elif "qwen" in args.model_name_or_path.lower():
                    input_ids = dev_data['input_ids'].to(device)
                    outputs = model.generate(input_ids=input_ids,
                                             max_new_tokens=args.max_length_generation,
                                             do_sample=args.do_sample,
                                             num_return_sequences=args.num_return_sequences,
                                             top_p=args.top_p,
                                             temperature=args.temperature,
                                             repetition_penalty=args.repetition_penalty,
                                             output_scores=not args.cot,
                                             return_dict_in_generate=not args.cot)
                else:
                    input_ids = dev_data['input_ids'].to(device)
                    attention_mask = dev_data['attention_mask'].to(device)
                    outputs = model.generate(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             max_new_tokens=args.max_length_generation,
                                             do_sample=args.do_sample,
                                             num_return_sequences=args.num_return_sequences,
                                             top_p=args.top_p,
                                             temperature=args.temperature,
                                             repetition_penalty=args.repetition_penalty,
                                             output_scores=not args.cot,
                                             return_dict_in_generate=not args.cot)

                # output processing and answer extraction
                if args.cot:
                    outputs = outputs['sequences'].tolist()[0][len(input_ids["input_ids"][0]):]
                    response = tokenizer.decode(outputs)
                    # response, _ = model.chat(tokenizer, dev_data['question'], history=dev_data['history'],
                    #                          do_sample=False, )
                    response = response.strip()
                    # ans, direct_extract = extract_cot_answer(dev_data, response)
                else:
                    logits = outputs['scores'][0].flatten()
                    pred = torch.tensor(
                                [
                                    logits[tokenizer.encode("A", add_special_tokens=False)[0]],
                                    logits[tokenizer.encode("B", add_special_tokens=False)[0]],
                                    logits[tokenizer.encode("C", add_special_tokens=False)[0]],
                                    logits[tokenizer.encode("D", add_special_tokens=False)[0]],
                                ]
                            ).argmax().detach().cpu().tolist()
                    pred = {0: "A", 1: "B", 2: "C", 3: "D"}[pred]
                    # correct = 1 if pred == label else 0
                    results[subject_name_key].append((dev_data['id'], dev_data['answer'], pred))

        # metrics calculation
        subject_mapping = json.load(open(os.path.join(RESOURCE_PATH, "eval", "ceval", "subject_mapping.json")))
        with open(output_filename, "w", encoding="utf-8") as w:
            result_dict = dict()
            acc_dict = dict()
            for subject_name_key, vals in results.items():
                if subject_name_key not in result_dict:
                    result_dict[subject_name_key] = dict()
                domain = subject_mapping[subject_name_key][2]
                if domain not in acc_dict:
                    acc_dict[domain] = {"ct": 0, "correct": 0}
                for id_, label, pred in vals:
                    result_dict[subject_name_key][str(id_)] = pred
                    acc_dict[domain]['correct'] += 1 if pred == label else 0
                    acc_dict[domain]['ct'] += 1
                    w.write(json.dumps({"subject_name_key": subject_name_key, "id": id_,
                                        "pred": pred, "label": label}, ensure_ascii=False)+"\n")

        # if submission file is not none, then there is no label to calculate accuracy
        if args.submission_filename is not None:
            json.dump(result_dict, open(os.path.join(args.output_dir, args.submission_filename), "w", encoding="utf-8"),
                      ensure_ascii=False)
            print_rank_0(f"Finished saving C-Eval Evaluation Result")
        else:
            ct = 0
            correct = 0
            for domain, val in acc_dict.items():
                ct += val['ct']
                correct += val['correct']
                print_rank_0(f"[C-Eval Evaluation Result] domain: {domain}, acc: {val['correct'] / val['ct']}")
            print_rank_0(f"[C-Eval Evaluation Result] total acc: {correct / ct}")
    elif args.task in ["mmlu"]:
        results = dict()
        with torch.no_grad():
            for dev_data in tqdm(dev_dataset, desc="MMLU Evaluation"):
                subject_name_key = dev_data['subject_name_key']
                if subject_name_key not in results:
                    results[subject_name_key] = list()
                if "chatglm" in args.model_name_or_path.lower():
                    logits_processor = LogitsProcessorList()
                    if "chatglm2" in args.model_name_or_path.lower():
                        class InvalidScoreLogitsProcessor(LogitsProcessor):
                            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                                if torch.isnan(scores).any() or torch.isinf(scores).any():
                                    scores.zero_()
                                    scores[..., 5] = 5e4
                                return scores
                    else:
                        class InvalidScoreLogitsProcessor(LogitsProcessor):
                            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                                if torch.isnan(scores).any() or torch.isinf(scores).any():
                                    scores.zero_()
                                    scores[..., 20005] = 5e4
                                return scores
                    logits_processor.append(InvalidScoreLogitsProcessor())
                    input_ids = dev_data['input_ids'].to(device)

                    outputs = model.generate(input_ids=input_ids,
                                             max_new_tokens=args.max_length_generation,
                                             do_sample=args.do_sample,
                                             num_return_sequences=args.num_return_sequences,
                                             top_p=args.top_p,
                                             temperature=args.temperature,
                                             repetition_penalty=args.repetition_penalty,
                                             logits_processor=logits_processor,
                                             output_scores=True,
                                             return_dict_in_generate=True)
                elif "qwen" in args.model_name_or_path.lower():
                    input_ids = dev_data['input_ids'].to(device)
                    outputs = model.generate(input_ids=input_ids,
                                             max_new_tokens=args.max_length_generation,
                                             do_sample=args.do_sample,
                                             num_return_sequences=args.num_return_sequences,
                                             top_p=args.top_p,
                                             temperature=args.temperature,
                                             repetition_penalty=args.repetition_penalty,
                                             output_scores=True,
                                             return_dict_in_generate=True)
                else:
                    input_ids = dev_data['input_ids'].to(device)
                    attention_mask = dev_data['attention_mask'].to(device)
                    outputs = model.generate(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             max_new_tokens=args.max_length_generation,
                                             do_sample=args.do_sample,
                                             num_return_sequences=args.num_return_sequences,
                                             top_p=args.top_p,
                                             temperature=args.temperature,
                                             repetition_penalty=args.repetition_penalty,
                                             output_scores=True,
                                             return_dict_in_generate=True)

                # output processing and answer extraction
                logits = outputs['scores'][0].flatten()
                pred = torch.tensor(
                    [
                        logits[tokenizer.encode("A", add_special_tokens=False)[0]],
                        logits[tokenizer.encode("B", add_special_tokens=False)[0]],
                        logits[tokenizer.encode("C", add_special_tokens=False)[0]],
                        logits[tokenizer.encode("D", add_special_tokens=False)[0]],
                    ]
                ).argmax().detach().cpu().tolist()
                pred = {0: "A", 1: "B", 2: "C", 3: "D"}[pred]
                # correct = 1 if pred == label else 0
                results[subject_name_key].append((dev_data['answer'], pred))

        # metrics calculation
        subject_mapping = json.load(open(os.path.join(RESOURCE_PATH, "eval", "mmlu", "subject_mapping.json")))
        with open(output_filename, "w", encoding="utf-8") as w:
            acc_dict = dict()
            for subject_name_key, vals in results.items():
                domain = subject_mapping[subject_name_key][1]
                if domain not in acc_dict:
                    acc_dict[domain] = {"ct": 0, "correct": 0}
                for label, pred in vals:
                    # result_dict[subject_name_key] = pred
                    acc_dict[domain]['correct'] += 1 if pred == label else 0
                    acc_dict[domain]['ct'] += 1
                    w.write(json.dumps({"subject_name_key": subject_name_key,
                                        "pred": pred, "label": label}, ensure_ascii=False)+"\n")
        ct = 0
        correct = 0
        for domain, val in acc_dict.items():
            ct += val['ct']
            correct += val['correct']
            print_rank_0(f"[MMLU Evaluation Result] domain: {domain}, acc: {val['correct'] / val['ct']}")
        print_rank_0(f"[MMLU Evaluation Result] total acc: {correct / ct}")
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

        print_rank_0(f"ppl={ct_acc/ct}")


if __name__ == "__main__":
    main()
