
import os
import json
import re
import random
import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.data import Dataset

from src.utils import logger
from src.utils.nlp_utils import clean_text


class DataCollatorReward:
    def __call__(self, data):
        batch = {"input_ids": torch.cat([f[0] for f in data] + [f[2] for f in data]),
                 "attention_mask": torch.cat([f[1] for f in data] + [f[3] for f in data]),
                 "labels": torch.tensor([0] * len(data) + [1] * len(data))}
        return batch


class PairwiseDataset(Dataset):
    def __init__(self, args, filename, tokenizer):
        self.pairs = self.load_dataset(filename)
        self.args = args
        self.tokenizer = tokenizer

        for k in range(5):
            logger.info(f"PairwiseDataset sample-{k}\n: {self.pairs[k]}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        prompt = pair["prompt"]
        chosen_answer = pair["chosen_answer"]
        rejected_answer = pair["rejected_answer"]
        chosen_encodings_dict = self.tokenizer(prompt + chosen_answer, max_length=self.args.max_length,
                                               truncation="longest_first", padding="max_length", return_tensors="pt")
        rejected_encodings_dict = self.tokenizer(prompt + rejected_answer, max_length=self.args.max_length,
                                                 truncation="longest_first", padding="max_length", return_tensors="pt")

        return {
            "chosen_input_ids": chosen_encodings_dict["input_ids"],
            "chosen_attention_mask": chosen_encodings_dict["attention_mask"],
            "rejected_input_ids": rejected_encodings_dict["input_ids"],
            "rejected_attention_mask": rejected_encodings_dict["attention_mask"],
        }

    @staticmethod
    def load_dataset(filename):
        discard = 0
        pairs = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                prompt = clean_text(item['prompt'])
                answers = item['answers']
                chosen_answer, rejected_answer = None, None
                for i in range(len(answers)-1):
                    answer_1 = clean_text(answers[i]["answer"])
                    answer_1_score = answers[i]["score"]
                    answer_2 = clean_text(answers[i+1]["answer"])
                    answer_2_score = answers[i+1]["score"]
                    if answer_1_score > answer_2_score:
                        chosen_answer = answer_1
                    rejected_answer = answer_2
                    # if (len(prompt) + len(rejected_answer) > max_length) or (len(prompt) + len(chosen_answer) > max_length):
                    #     discard += 1
                    # else:
                    if chosen_answer is not None:
                        pair = {
                            "prompt": prompt,
                            "chosen_answer": chosen_answer,
                            "rejected_answer": rejected_answer
                        }
                        pairs.append(pair)
        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return pairs


class SFTDataset(Dataset):
    def __init__(self, args, filename, tokenizer):
        dataset = self.load_dataset(filename)
        self.post_list = dataset
        # self.post_list = []
        # for sample in dataset:
        #     self.post_list.append((sample["prompt"], "????????????:" + sample["label"]))

        self.tokenizer = tokenizer
        self.args = args

        for k in range(5):
            logger.info(f"SFTDataset sample-{k}\n: {dataset[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        data = self.post_list[idx]
        prompt = data['prompt']
        label = data['label']
        encoded_dict = self.tokenizer(prompt + label, max_length=self.args.max_length,
                                      padding="max_length", truncation="longest_first", return_tensors="pt")

        return {
            "input_ids": encoded_dict["input_ids"],
            "attention_mask": encoded_dict["attention_mask"],
            "labels": encoded_dict['input_ids'],
        }

    @staticmethod
    def load_dataset(filename):
        discard = 0
        datasets = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                prompt = clean_text(item['prompt'])
                label = clean_text(item['answers'][0]['answer'])

                if len(prompt) <= 0 or len(label) <= 0:
                    continue
                datasets.append({"prompt": prompt, "label": label})
        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class RLHFDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):

        # dataset = self.load_dataset(filename)
        self.post_list = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        for k in range(5):
            logger.info(f"TLDRDataset sample-{k}\n: {dataset[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        sample = self.post_list[idx]
        encodings_dict = self.tokenizer(sample["prompt"], "????????????:" + sample["label"], max_length=self.max_length,
                                        truncation="longest_first", return_tensors="pt")
        text = self.tokenizer.decode(encodings_dict['input_ids'], skip_special_tokens=True).strip()

        return text

    @staticmethod
    def load_dataset(filename, max_length):
        discard = 0
        datasets = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc="Load Dataset"):
                item = json.loads(line)
                prompt = clean_text(item['title'] if len(item['title']) > len(item['desc']) else item['desc'])
                label = clean_text(item['answer'])

                if len(prompt) + len(label) > max_length:
                    discard += 1
                else:
                    datasets.append({"prompt": prompt, "label": label})
                # if i-discard+1 == max_samples:
                #     logger.info(f"File: {path}/web_text_zh_{split}.json Num of over-length: {discard}")
                #     return datasets
        logger.info(f"Finish loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class ComparisonDataset(Dataset):
    def __init__(self, comparison_path, tokenizer, max_length):
        with open(comparison_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f]

        self.tokenizer = tokenizer
        self.post_list = []
        self.summaries_0 = []
        self.summaries_1 = []
        self.labels = []
        self.max_length = max_length

        def make_text(post, summarize):
            return f"SUBREDDIT: r/{post['subreddit']}\nTITLE: {post['title']}\nPOST: {post['post']}\nTL;DR: {summarize}"

        for sample in dataset:  # chosen summary is always the first one
            self.post_list.append(sample["info"]["post"])
            # NOTE: The chosen summary is always the first one, i.e. `sample["summaries"][0]`
            if sample["choice"] == 0:
                self.summaries_0.append(make_text(sample["info"], sample["summaries"][0]["text"]))
                self.summaries_1.append(make_text(sample["info"], sample["summaries"][1]["text"]))
            else:
                self.summaries_0.append(make_text(sample["info"], sample["summaries"][1]["text"]))
                self.summaries_1.append(make_text(sample["info"], sample["summaries"][0]["text"]))
            self.labels.append(0)

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        summ0 = self.summaries_0[idx]
        summ1 = self.summaries_1[idx]
        encodings_dict = self.tokenizer(
            [summ0, summ1],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attention_mask = torch.tensor(encodings_dict["attention_mask"])
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class AllSummDataset(Dataset):
    def __init__(self, train_path, tokenizer, split, max_length=1024):
        df = pd.read_parquet(train_path)
        if split == "valid":
            df = df.sample(n=5000)
        self.summarizes = []
        for i, row in df.iterrows():
            self.summarizes.append(f"Summarize: {row['text']}. TL;DR: {row['summary']}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.summarizes)

    def __getitem__(self, idx):
        txt = self.summarizes[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
        }


class OCNLIDataset(Dataset):
    def __init__(self, args, eval_filename, tokenizer, train_filename=None):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'entailment': 'Yes', 'neutral': 'Maybe', 'contradiction': 'No'}

        dataset = self.load_dataset(eval_filename)
        if train_filename is not None:
            self.labelled_list = self.load_dataset(eval_filename)
        self.post_list = dataset

        for k in range(5):
            logger.info(f"OCNLIDataset sample-{k}\n: {dataset[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        data = self.post_list[idx]
        prompt = data['prompt']
        label = data['label']

        # Few-Shot example construction
        if hasattr(self, "labelled_list"):
            examples = random.sample(self.labelled_list, min(len(self.labelled_list), self.args.max_few_shot))
            prompts = []
            prompt_tokens = self.tokenizer.tokenize(prompt)
            for example in examples:
                example_prompt = example['prompt']
                exmample_tokens = self.tokenizer.tokenize(example_prompt+"\n")
                if len(exmample_tokens) + len(prompt_tokens) + 2 > self.args.max_length:
                    break
                else:
                    prompts.append(example_prompt)
                    prompt_tokens.extend(exmample_tokens)
            prompts.append(prompt)
            prompt = "\n".join(prompts)

        encoded_dict = self.tokenizer(prompt, max_length=self.args.max_length,
                                      padding="max_length", truncation="longest_first", return_tensors="pt")

        return {
            "input_ids": encoded_dict["input_ids"],
            "attention_mask": encoded_dict["attention_mask"],
            "labels": encoded_dict["input_ids"],
            "label_str": label
        }

    def load_dataset(self, filename):
        discard = 0
        datasets = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                s1 = item['sentence1']
                s2 = item['sentence2']
                label = item['label']
                # ?????????????????????????????????
                if label == "-":
                    continue
                for l in self.label_dict.values():
                    prompt = f'{s1}?{l}???{s2}'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": self.label_dict[label]})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class CMNLIDataset(Dataset):
    def __init__(self, args, eval_filename, tokenizer, train_filename=None):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'entailment': 'Yes', 'neutral': 'Maybe', 'contradiction': 'No'}

        dataset = self.load_dataset(eval_filename)
        if train_filename is not None:
            self.labelled_list = self.load_dataset(eval_filename)
        self.post_list = dataset

        for k in range(5):
            logger.info(f"CMNLIDataset sample-{k}\n: {dataset[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        data = self.post_list[idx]
        prompt = data['prompt']
        label = data['label']

        # Few-Shot example construction
        if hasattr(self, "labelled_list"):
            examples = random.sample(self.labelled_list, min(len(self.labelled_list), self.args.max_few_shot))
            prompts = []
            prompt_tokens = self.tokenizer.tokenize(prompt)
            for example in examples:
                example_prompt = example['prompt']
                exmample_tokens = self.tokenizer.tokenize(example_prompt+"\n")
                if len(exmample_tokens) + len(prompt_tokens) + 2 > self.args.max_length:
                    break
                else:
                    prompts.append(example_prompt)
                    prompt_tokens.extend(exmample_tokens)
            prompts.append(prompt)
            prompt = "\n".join(prompts)

        encoded_dict = self.tokenizer(prompt, max_length=self.args.max_length,
                                      padding="max_length", truncation="longest_first", return_tensors="pt")
        # label_dict = self.tokenizer(label, max_length=self.args.max_length, add_special_tokens=False,
        #                             return_attention_mask=False, return_token_type_ids=False, return_tensors="pt")

        return {
            "input_ids": encoded_dict["input_ids"],
            "attention_mask": encoded_dict["attention_mask"],
            "labels": encoded_dict["input_ids"],
            "label_str": label
        }

    def load_dataset(self, filename):
        discard = 0
        datasets = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                s1 = item['sentence1']
                s2 = item['sentence2']
                label = item['label']
                # ?????????????????????????????????
                if label == "-":
                    continue
                for l in self.label_dict.values():
                    prompt = f'{s1}?{l}???{s2}'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": self.label_dict[label]})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class CHIDDataset(Dataset):
    def __init__(self, args, eval_filename, tokenizer, train_filename=None):
        self.tokenizer = tokenizer
        self.args = args

        self.idiom_dict = self.load_idiom_dict()
        dataset = self.load_dataset(eval_filename)
        if train_filename is not None:
            self.labelled_list = self.load_dataset(eval_filename)
        self.post_list = dataset

        for k in range(5):
            logger.info(f"CHIDDataset sample-{k}\n: {dataset[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        data = self.post_list[idx]
        prompt = data['prompt']
        label = data['label']
        candidates = data['candidates']

        # Few-Shot example construction
        if hasattr(self, "labelled_list"):
            examples = random.sample(self.labelled_list, min(len(self.labelled_list), self.args.max_few_shot))
            prompts = []
            prompt_tokens = self.tokenizer.tokenize(prompt)
            for example in examples:
                example_prompt = example['prompt']
                exmample_tokens = self.tokenizer.tokenize(example_prompt+"\n")
                if len(exmample_tokens) + len(prompt_tokens) + 2 > self.args.max_length:
                    break
                else:
                    prompts.append(example_prompt)
                    prompt_tokens.extend(exmample_tokens)
            prompts.append(prompt)
            prompt = "\n".join(prompts)

        encoded_dict = self.tokenizer(prompt, max_length=self.args.max_length,
                                      padding="max_length", truncation="longest_first", return_tensors="pt")
        # label_dict = self.tokenizer(label, max_length=self.args.max_length, add_special_tokens=False,
        #                             return_attention_mask=False, return_token_type_ids=False, return_tensors="pt")

        return {
            "input_ids": encoded_dict["input_ids"],
            "attention_mask": encoded_dict["attention_mask"],
            "labels": encoded_dict["input_ids"],
            "label_str": label,
            "candidates": candidates
        }

    def load_dataset(self, filename):
        discard = 0
        datasets = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                candidates = item['candidates']
                contents = item['content']
                for content in contents:
                    for idiom in re.findall(r"#idiom\d+#", content):
                        label = candidates[self.idiom_dict[idiom]]
                        for candidate in candidates:
                            prompt = content.replace(idiom, candidate)
                            if len(prompt) <= 0:
                                continue
                            datasets.append({"prompt": prompt, "label": label, "candidates": candidates})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets

    def load_idiom_dict(self):
        idiom_dict = json.load(open(os.path.join(self.args.data_dir, "dev_answer.json"), "r", encoding="utf-8"))
        idiom_dict.update(json.load(open(os.path.join(self.args.data_dir, "train_answer.json"), "r", encoding="utf-8")))

        logger.info(f"Finished loading idiom dict")

        return idiom_dict


class CMRCDataset(Dataset):
    def __init__(self, args, eval_filename, tokenizer, train_filename=None):
        self.tokenizer = tokenizer
        self.args = args

        dataset = self.load_dataset(eval_filename)
        if train_filename is not None:
            self.labelled_list = self.load_dataset(eval_filename)
        self.post_list = dataset

        for k in range(5):
            logger.info(f"CMRCDataset sample-{k}\n: {dataset[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        data = self.post_list[idx]
        prompt = data['prompt']
        label = data['label']

        # Few-Shot example construction
        if hasattr(self, "labelled_list"):
            examples = random.sample(self.labelled_list, min(len(self.labelled_list), self.args.max_few_shot))
            prompts = []
            prompt_tokens = self.tokenizer.tokenize(prompt)
            for example in examples:
                example_prompt = example['prompt']
                exmample_tokens = self.tokenizer.tokenize(example_prompt+"\n")
                if len(exmample_tokens) + len(prompt_tokens) + 2 > self.args.max_length:
                    break
                else:
                    prompts.append(example_prompt)
                    prompt_tokens.extend(exmample_tokens)
            prompts.append(prompt)
            prompt = "\n".join(prompts)

        encoded_dict = self.tokenizer(prompt, max_length=self.args.max_length,
                                      padding="max_length", truncation="longest_first", return_tensors="pt")

        return {
            "input_ids": encoded_dict["input_ids"],
            "attention_mask": encoded_dict["attention_mask"],
            "labels": encoded_dict["input_ids"],
            "label_str": label
        }

    def load_dataset(self, filename):
        discard = 0
        datasets = []
        data = json.load(open(filename, "r", encoding="utf-8"))
        for paragraphs in data['data']:
            for paragraph in paragraphs['paragraphs']:
                context = paragraph['context']
                for qs in paragraph['qas']:
                    question = qs['question']
                    answers = []
                    [answers.append(answer) for answer in qs['answers'] if answer not in answers]
                    prompt_template = "???????????????{context}\n??????{question}\n??????"
                    prompt = prompt_template.format(context=context, question=question)
                    if len(prompt) <= 0:
                        continue
                    # if len(prompt) > self.args.max_length:
                    #     idx = len(prompt) - self.args.max_length
                    #     prompt = prompt_template.format(context=context[:-idx], question=question)
                    datasets.append({"prompt": prompt, "label": answers})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class CLUEWSCDataset(Dataset):
    def __init__(self, args, eval_filename, tokenizer, train_filename=None):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'true': '1', 'false': '0'}

        dataset = self.load_dataset(eval_filename)
        if train_filename is not None:
            self.labelled_list = self.load_dataset(eval_filename)
        self.post_list = dataset

        for k in range(5):
            logger.info(f"CLUEWSCDataset sample-{k}\n: {dataset[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        data = self.post_list[idx]
        prompt = data['prompt']
        label = data['label']

        # Few-Shot example construction
        if hasattr(self, "labelled_list"):
            examples = random.sample(self.labelled_list, min(len(self.labelled_list), self.args.max_few_shot))
            prompts = []
            prompt_tokens = self.tokenizer.tokenize(prompt)
            for example in examples:
                example_prompt = example['prompt']
                exmample_tokens = self.tokenizer.tokenize(example_prompt+"\n")
                if len(exmample_tokens) + len(prompt_tokens) + 2 > self.args.max_length:
                    break
                else:
                    prompts.append(example_prompt)
                    prompt_tokens.extend(exmample_tokens)
            prompts.append(prompt)
            prompt = "\n".join(prompts)

        encoded_dict = self.tokenizer(prompt, max_length=self.args.max_length,
                                      padding="max_length", truncation="longest_first", return_tensors="pt")
        # label_dict = self.tokenizer(label, max_length=self.args.max_length, add_special_tokens=False,
        #                             return_attention_mask=False, return_token_type_ids=False, return_tensors="pt")

        return {
            "input_ids": encoded_dict["input_ids"],
            "attention_mask": encoded_dict["attention_mask"],
            "labels": encoded_dict["input_ids"],
            "label_str": label,
        }

    def load_dataset(self, filename):
        discard = 0
        datasets = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                text = item['text']
                span2_index = item['target']['span2_index']
                span2_text = item['target']['span2_text']
                span1_text = item['target']['span1_text']
                label = self.label_dict[item['label']]
                prompt = text[:span2_index] + span1_text + text[span2_index+len(span2_text):]
                if len(prompt) <= 0:
                    continue
                datasets.append({"prompt": prompt, "label": label})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class C3Dataset(Dataset):
    def __init__(self, args, eval_filename, tokenizer, train_filename=None):
        self.tokenizer = tokenizer
        self.args = args

        dataset = self.load_dataset(eval_filename)
        if train_filename is not None:
            self.labelled_list = self.load_dataset(eval_filename)
        self.post_list = dataset

        for k in range(5):
            logger.info(f"C3Dataset sample-{k}\n: {dataset[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        data = self.post_list[idx]
        prompt = data['prompt']
        label = data['label']
        candidates = data['candidates']

        # Few-Shot example construction
        if hasattr(self, "labelled_list"):
            examples = random.sample(self.labelled_list, min(len(self.labelled_list), self.args.max_few_shot))
            prompts = []
            prompt_tokens = self.tokenizer.tokenize(prompt)
            for example in examples:
                example_prompt = example['prompt']
                exmample_tokens = self.tokenizer.tokenize(example_prompt+"\n")
                if len(exmample_tokens) + len(prompt_tokens) + 2 > self.args.max_length:
                    break
                else:
                    prompts.append(example_prompt)
                    prompt_tokens.extend(exmample_tokens)
            prompts.append(prompt)
            prompt = "\n".join(prompts)


        encoded_dict = self.tokenizer(prompt, max_length=self.args.max_length,
                                      padding="max_length", truncation="longest_first", return_tensors="pt")
        # label_dict = self.tokenizer(label, max_length=self.args.max_length, add_special_tokens=False,
        #                             return_attention_mask=False, return_token_type_ids=False, return_tensors="pt")

        return {
            "input_ids": encoded_dict["input_ids"],
            "attention_mask": encoded_dict["attention_mask"],
            "labels": encoded_dict["input_ids"],
            "label_str": label,
            "candidates": candidates
        }

    def load_dataset(self, filename):
        discard = 0
        datasets = []

        data = json.load(open(filename, "r", encoding="utf-8"))
        for i, d in enumerate(data):
            context = "".join(d[0])
            for qs in d[1]:
                question = qs['question']
                choices = qs['choice']
                choices_padded = [choices[i] if i < len(choices) else f"test{i}" for i in range(4)]
                answer = qs['answer']
                for choice in choices:
                    prompt = f"???: {question}\n???:{choice}\n?????????????????????: {context}"
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": answer, "candidates": choices_padded})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class AFQMCDataset(Dataset):
    def __init__(self, args, eval_filename, tokenizer, train_filename=None):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'0': '??????', '1': '??????'}

        dataset = self.load_dataset(eval_filename)
        if train_filename is not None:
            self.labelled_list = self.load_dataset(eval_filename)
        self.post_list = dataset

        for k in range(5):
            logger.info(f"AFQMCDataset sample-{k}\n: {dataset[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        data = self.post_list[idx]
        prompt = data['prompt']
        label = data['label']

        # Few-Shot example construction
        if hasattr(self, "labelled_list"):
            examples = random.sample(self.labelled_list, min(len(self.labelled_list), self.args.max_few_shot))
            prompts = []
            prompt_tokens = self.tokenizer.tokenize(prompt)
            for example in examples:
                example_prompt = example['prompt']
                exmample_tokens = self.tokenizer.tokenize(example_prompt+"\n")
                if len(exmample_tokens) + len(prompt_tokens) + 2 > self.args.max_length:
                    break
                else:
                    prompts.append(example_prompt)
                    prompt_tokens.extend(exmample_tokens)
            prompts.append(prompt)
            prompt = "\n".join(prompts)

        encoded_dict = self.tokenizer(prompt, max_length=self.args.max_length,
                                      padding="max_length", truncation="longest_first", return_tensors="pt")
        # label_dict = self.tokenizer(label, max_length=self.args.max_length, add_special_tokens=False,
        #                             return_attention_mask=False, return_token_type_ids=False, return_tensors="pt")

        return {
            "input_ids": encoded_dict["input_ids"],
            "attention_mask": encoded_dict["attention_mask"],
            "labels": encoded_dict["input_ids"],
            "label_str": label
        }

    def load_dataset(self, filename):
        discard = 0
        datasets = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                s1 = item['sentence1']
                s2 = item['sentence2']
                label = self.label_dict[item['label']]
                for l in self.label_dict.values():
                    prompt = f'????????????????????????{l}:{s1}???{s2}'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": label})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class CSLDataset(Dataset):
    def __init__(self, args, eval_filename, tokenizer, train_filename=None):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'0': '??????', '1': '???'}

        dataset = self.load_dataset(eval_filename)
        if train_filename is not None:
            self.labelled_list = self.load_dataset(eval_filename)
        self.post_list = dataset

        for k in range(5):
            logger.info(f"CSLDataset sample-{k}\n: {dataset[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        data = self.post_list[idx]
        prompt = data['prompt']
        label = data['label']

        # Few-Shot example construction
        if hasattr(self, "labelled_list"):
            examples = random.sample(self.labelled_list, min(len(self.labelled_list), self.args.max_few_shot))
            prompts = []
            prompt_tokens = self.tokenizer.tokenize(prompt)
            for example in examples:
                example_prompt = example['prompt']
                exmample_tokens = self.tokenizer.tokenize(example_prompt+"\n")
                if len(exmample_tokens) + len(prompt_tokens) + 2 > self.args.max_length:
                    break
                else:
                    prompts.append(example_prompt)
                    prompt_tokens.extend(exmample_tokens)
            prompts.append(prompt)
            prompt = "\n".join(prompts)

        encoded_dict = self.tokenizer(prompt, max_length=self.args.max_length,
                                      padding="max_length", truncation="longest_first", return_tensors="pt")
        # label_dict = self.tokenizer(label, max_length=self.args.max_length, add_special_tokens=False,
        #                             return_attention_mask=False, return_token_type_ids=False, return_tensors="pt")

        return {
            "input_ids": encoded_dict["input_ids"],
            "attention_mask": encoded_dict["attention_mask"],
            "labels": encoded_dict["input_ids"],
            "label_str": label
        }

    def load_dataset(self, filename):
        discard = 0
        datasets = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                abstract = item['abst']
                keyword = "+".join(item['keyword'])
                label = self.label_dict[item['label']]
                for l in self.label_dict.values():
                    prompt = f'??????:{abstract}????????????:{keyword}{l}???????????????'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": label})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class IFLYTEKDataset(Dataset):
    def __init__(self, args, eval_filename, tokenizer, train_filename=None):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'0': '??????', '1': '????????????', '2': '??????WIFI', '3': '??????', '4': '????????????', '5': '????????????', '6': '??????', '7': '??????', '8': '????????????', '9': '??????', '10': '????????????', '11': '?????????', '12': '??????', '13': '??????', '14': '??????', '15': '????????????', '16': '????????????', '17': '????????????', '18': '?????????', '19': '????????????', '20': '????????????', '21': '????????????', '22': '??????', '23': 'MOBA', '24': '????????????', '25': '????????????', '26': '????????????', '27': '????????????', '28': '????????????', '29': '????????????', '30': '????????????', '31': '????????????', '32': '????????????', '33': '????????????', '34': '??????', '35': '??????', '36': '??????', '37': '??????', '38': '??????', '39': '????????????', '40': '??????', '41': '??????', '42': '??????', '43': '????????????', '44': '??????', '45': '??????', '46': '??????', '47': '?????????', '48': '??????', '49': '??????', '50': '??????', '51': 'K???', '52': '??????', '53': '?????????', '54': '??????', '55': '?????????', '56': '??????', '57': '????????????', '58': '????????????', '59': '????????????', '60': '??????', '61': '??????(?????????)', '62': '????????????', '63': '????????????', '64': '??????', '65': '??????', '66': '??????', '67': '????????????', '68': '????????????', '69': '??????', '70': '??????', '71': '????????????', '72': '??????', '73': '??????', '74': '??????', '75': '????????????', '76': '????????????', '77': '????????????', '78': '????????????', '79': '??????', '80': '??????', '81': '????????????', '82': '????????????', '83': '????????????', '84': '????????????', '85': '????????????', '86': '????????????', '87': '????????????', '88': '??????', '89': '?????????', '90': '????????????', '91': '????????????', '92': '??????', '93': '??????', '94': '??????', '95': '??????', '96': '??????', '97': '??????', '98': '??????', '99': '??????', '100': '??????', '101': '????????????', '102': '????????????', '103': '??????', '104': '??????', '105': '??????', '106': '??????', '107': '??????', '108': '??????', '109': '????????????', '110': '????????????', '111': '????????????', '112': '??????', '113': '??????', '114': '????????????', '115': '??????', '116': '??????', '117': '??????', '118': '??????'}

        dataset = self.load_dataset(eval_filename)
        if train_filename is not None:
            self.labelled_list = self.load_dataset(eval_filename)
        self.post_list = dataset

        for k in range(5):
            logger.info(f"IFLYTEKDataset sample-{k}\n: {dataset[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        data = self.post_list[idx]
        prompt = data['prompt']
        label = data['label']
        candidates = data['candidates']

        # Few-Shot example construction
        if hasattr(self, "labelled_list"):
            examples = random.sample(self.labelled_list, min(len(self.labelled_list), self.args.max_few_shot))
            prompts = []
            prompt_tokens = self.tokenizer.tokenize(prompt)
            for example in examples:
                example_prompt = example['prompt']
                exmample_tokens = self.tokenizer.tokenize(example_prompt+"\n")
                if len(exmample_tokens) + len(prompt_tokens) + 2 > self.args.max_length:
                    break
                else:
                    prompts.append(example_prompt)
                    prompt_tokens.extend(exmample_tokens)
            prompts.append(prompt)
            prompt = "\n".join(prompts)

        encoded_dict = self.tokenizer(prompt, max_length=self.args.max_length,
                                      padding="max_length", truncation="longest_first", return_tensors="pt")
        # label_dict = self.tokenizer(label, max_length=self.args.max_length, add_special_tokens=False,
        #                             return_attention_mask=False, return_token_type_ids=False, return_tensors="pt")

        return {
            "input_ids": encoded_dict["input_ids"],
            "attention_mask": encoded_dict["attention_mask"],
            "labels": encoded_dict["input_ids"],
            "label_str": label,
            "candidates": candidates
        }

    def load_dataset(self, filename):
        discard = 0
        datasets = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                content = item['sentence']
                label = item['label_des']
                # randomly sample 3 categories as negative sample
                labels = set(self.label_dict.values())
                labels.remove(label)
                candidates = [label] + random.sample(labels, 3)
                for l in candidates:
                    prompt = f'????????????{l}???????????????:{content}'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": label, "candidates": candidates})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class TNEWSDataset(Dataset):
    def __init__(self, args, eval_filename, tokenizer, train_filename=None):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'100': '??????',
                           '101': '??????',
                           '102': '??????',
                           '103': '??????',
                           '104': '??????',
                           '106': '??????',
                           '107': '??????',
                           '108': '??????',
                           '109': '??????',
                           '110': '??????',
                           '112': '??????',
                           '113': '??????',
                           '114': '??????',
                           '115': '??????',
                           '116': '??????'}

        dataset = self.load_dataset(eval_filename)
        if train_filename is not None:
            self.labelled_list = self.load_dataset(eval_filename)
        self.post_list = dataset

        for k in range(5):
            logger.info(f"TNEWSDataset sample-{k}\n: {dataset[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        data = self.post_list[idx]
        prompt = data['prompt']
        label = data['label']
        candidates = data['candidates']

        # Few-Shot example construction
        if hasattr(self, "labelled_list"):
            examples = random.sample(self.labelled_list, min(len(self.labelled_list), self.args.max_few_shot))
            prompts = []
            prompt_tokens = self.tokenizer.tokenize(prompt)
            for example in examples:
                example_prompt = example['prompt']
                exmample_tokens = self.tokenizer.tokenize(example_prompt+"\n")
                if len(exmample_tokens) + len(prompt_tokens) + 2 > self.args.max_length:
                    break
                else:
                    prompts.append(example_prompt)
                    prompt_tokens.extend(exmample_tokens)
            prompts.append(prompt)
            prompt = "\n".join(prompts)

        encoded_dict = self.tokenizer(prompt, max_length=self.args.max_length,
                                      padding="max_length", truncation="longest_first", return_tensors="pt")
        # label_dict = self.tokenizer(label, max_length=self.args.max_length, add_special_tokens=False,
        #                             return_attention_mask=False, return_token_type_ids=False, return_tensors="pt")

        return {
            "input_ids": encoded_dict["input_ids"],
            "attention_mask": encoded_dict["attention_mask"],
            "labels": encoded_dict["input_ids"],
            "label_str": label,
            "candidates": candidates
        }

    def load_dataset(self, filename):
        discard = 0
        datasets = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                content = item['sentence']
                label = self.label_dict[item['label']]
                # randomly sample 3 categories as negative sample
                labels = set(self.label_dict.values())
                labels.remove(label)
                candidates = [label] + random.sample(labels, 3)
                for l in candidates:
                    prompt = f'????????????{l}?????????:{content}'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": label, "candidates": candidates})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets