
import os
import json
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
        self.pairs = self.load_dataset(filename, args.max_length)
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
    def load_dataset(filename, max_length):
        discard = 0
        pairs = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                prompt = clean_text(item['prompt'])
                answers = item['answers']
                chosen_answer = answers[0]["answer"]
                for answer in answers[1:]:
                    rejected_answer = answer["answer"]
                    # if (len(prompt) + len(rejected_answer) > max_length) or (len(prompt) + len(chosen_answer) > max_length):
                    #     discard += 1
                    # else:
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
        dataset = self.load_dataset(filename, args.max_length)
        self.post_list = dataset
        # self.post_list = []
        # for sample in dataset:
        #     self.post_list.append((sample["prompt"], "模型回答:" + sample["label"]))

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
    def load_dataset(filename, max_length):
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
        encodings_dict = self.tokenizer(sample["prompt"], "模型回答:" + sample["label"], max_length=self.max_length,
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
    def __init__(self, args, filename, tokenizer):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'entailment': 'Yes', 'neutral': 'Maybe', 'contradiction': 'No'}

        dataset = self.load_dataset(filename, args.max_length)
        self.post_list = dataset

        for k in range(5):
            logger.info(f"OCNLIDataset sample-{k}\n: {dataset[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        data = self.post_list[idx]
        prompt = data['prompt']
        label = data['label']
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

    def load_dataset(self, filename, max_length):
        discard = 0
        datasets = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                s1 = item['sentence1']
                s2 = item['sentence2']
                label = item['label']
                # 标注结果有冲突，则忽略
                if label == "-":
                    continue
                for l in self.label_dict.values():
                    prompt = f'{s1}?{l}，{s2}'
                    if len(prompt) <= 0 or len(label) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": self.label_dict[label]})
        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets
