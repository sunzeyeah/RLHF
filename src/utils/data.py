
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
    def __init__(self, filename, tokenizer, max_length):
        pairs = self.create_comparison_dataset(filename)

        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        num_skip = 0

        for pair in tqdm(pairs):
            chosen_prompt, chosen_summary = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(chosen_prompt, chosen_summary, max_length=max_length,
                                              truncation="longest_first", padding="max_length", return_tensors="pt")
            rejected_prompt, rejected_summary = pair["chosen"], pair["rejected"]
            rejected_encodings_dict = tokenizer(rejected_prompt, rejected_summary, max_length=max_length,
                                                truncation="longest_first", padding="max_length", return_tensors="pt")

            if torch.all(torch.eq(chosen_encodings_dict["input_ids"], rejected_encodings_dict["input_ids"])).item():
                num_skip += 1
                continue

            self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
            self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
            self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
            self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

        logger.info(f"PairwiseDataset # skipped instances: {num_skip}")

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )

    @staticmethod
    def create_comparison_dataset(filename):
        dataset = torch.load(filename)
        for k in range(5):
            logger.info(f"PairwiseDataset sample-{k}\n: {dataset[-k-1]}")

        pairs = []
        for sample in tqdm(dataset):
            prompt = sample["prompt"]
            chosen_summary = sample["chosen"]
            rejected_summary = sample["rejected"]
            if chosen_summary == rejected_summary:
                continue
            if len(chosen_summary) < 5 or len(rejected_summary) < 5:
                continue
            pair = {"chosen": [prompt, chosen_summary],
                    "rejected": [prompt, rejected_summary]}
            pairs.append(pair)
        return pairs


class SFTDataset(Dataset):
    def __init__(self, args, filename, tokenizer):
        dataset = self.load_dataset(filename, args.max_length)
        # self.post_list = dataset
        self.post_list = []
        for sample in dataset:
            self.post_list.append((sample["prompt"], "模型回答:" + sample["label"]))

        self.tokenizer = tokenizer
        self.args = args

        for k in range(5):
            logger.info(f"SFTDataset sample-{k}\n: {dataset[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        prompt, label = self.post_list[idx]
        encoded_dict = self.tokenizer(prompt, label, max_length=self.args.max_length,
                                      padding="max_length", truncation="longest_first", return_tensors="pt")

        return {
            "input_ids": encoded_dict["input_ids"],
            "attention_mask": encoded_dict["attention_mask"],
            "labels": encoded_dict['input_ids'],
        }
        # prompt = prompt[:(self.args.max_length_prompt-6)]
        # prompt += self.tokenizer.sep_token + "模型回答:"
        # encoded_prompt = self.tokenizer(prompt, add_special_tokens=False, max_length=self.args.max_length_prompt,
        #                                 padding="max_length", truncation="longest_first", return_tensors="pt")
        # encoded_label = self.tokenizer(label, max_length=self.args.max_length_label,
        #                                padding="max_length", truncation="longest_first", return_tensors="pt")
        #
        # return {
        #     "input_ids": torch.cat((encoded_prompt["input_ids"], encoded_label['input_ids']), axis=1),
        #     "attention_mask": torch.cat((encoded_prompt["attention_mask"], encoded_label['attention_mask']), axis=1),
        #     "labels": encoded_label['input_ids'],
        # }

    @staticmethod
    def load_dataset(filename, max_length):
        discard = 0
        datasets = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                prompt = clean_text(item['title'] if len(item['title']) > len(item['desc']) else item['desc'])
                label = clean_text(item['answer'])

                if len(prompt) + len(label) > max_length:
                    discard += 1
                else:
                    datasets.append({"prompt": prompt, "label": label})
                # if split == "valid" and i == 2000:
                #     logger.info(f"File: {path}/web_text_zh_{split}_small.json, Num of over-length: {discard}")
                #     return datasets
        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class TLDRDataset(Dataset):
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

