
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
        dataset = self.load_dataset(filename)
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

        dataset = self.load_dataset(filename)
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
                # 标注结果有冲突，则忽略
                if label == "-":
                    continue
                for l in self.label_dict.values():
                    prompt = f'{s1}?{l}，{s2}'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": self.label_dict[label]})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class CMNLIDataset(Dataset):
    def __init__(self, args, filename, tokenizer):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'entailment': 'Yes', 'neutral': 'Maybe', 'contradiction': 'No'}

        dataset = self.load_dataset(filename)
        self.post_list = dataset

        for k in range(5):
            logger.info(f"CMNLIDataset sample-{k}\n: {dataset[k]}")

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

    def load_dataset(self, filename):
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
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": self.label_dict[label]})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class CHIDDataset(Dataset):
    def __init__(self, args, filename, tokenizer):
        self.tokenizer = tokenizer
        self.args = args

        self.idiom_dict = self.load_idiom_dict()
        dataset = self.load_dataset(filename)
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

        logger.info(f"Finished loading idiom dict")

        return idiom_dict


class CMRCDataset(Dataset):
    def __init__(self, args, filename, tokenizer):
        self.tokenizer = tokenizer
        self.args = args

        dataset = self.load_dataset(filename)
        self.post_list = dataset

        for k in range(5):
            logger.info(f"CMRCDataset sample-{k}\n: {dataset[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        data = self.post_list[idx]
        prompt = data['prompt']
        label = data['label']
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
                    prompt_template = "阅读文章：{context}\n问：{question}\n答："
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
    def __init__(self, args, filename, tokenizer):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'true': '1', 'false': '0'}
        dataset = self.load_dataset(filename)
        self.post_list = dataset

        for k in range(5):
            logger.info(f"CLUEWSCDataset sample-{k}\n: {dataset[k]}")

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
    def __init__(self, args, filename, tokenizer):
        self.tokenizer = tokenizer
        self.args = args

        dataset = self.load_dataset(filename)
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
                    prompt = f"问: {question}\n答:{choice}\n该答案来自对话: {context}"
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": answer, "candidates": choices_padded})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class AFQMCDataset(Dataset):
    def __init__(self, args, filename, tokenizer):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'0': '不同', '1': '相同'}

        dataset = self.load_dataset(filename)
        self.post_list = dataset

        for k in range(5):
            logger.info(f"AFQMCDataset sample-{k}\n: {dataset[k]}")

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
                    prompt = f'下面两个句子语义{l}:{s1}。{s2}'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": label})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class CSLDataset(Dataset):
    def __init__(self, args, filename, tokenizer):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'0': '不是', '1': '是'}

        dataset = self.load_dataset(filename)
        self.post_list = dataset

        for k in range(5):
            logger.info(f"CSLDataset sample-{k}\n: {dataset[k]}")

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
                    prompt = f'摘要:{abstract}，关键词:{keyword}{l}真实关键词'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": label})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class IFLYTEKDataset(Dataset):
    def __init__(self, args, filename, tokenizer):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'0': '打车', '1': '地图导航', '2': '免费WIFI', '3': '租车', '4': '同城服务', '5': '快递物流', '6': '婚庆', '7': '家政', '8': '公共交通', '9': '政务', '10': '社区服务', '11': '薅羊毛', '12': '魔幻', '13': '仙侠', '14': '卡牌', '15': '飞行空战', '16': '射击游戏', '17': '休闲益智', '18': '动作类', '19': '体育竞技', '20': '棋牌中心', '21': '经营养成', '22': '策略', '23': 'MOBA', '24': '辅助工具', '25': '约会社交', '26': '即时通讯', '27': '工作社交', '28': '论坛圈子', '29': '婚恋社交', '30': '情侣社交', '31': '社交工具', '32': '生活社交', '33': '微博博客', '34': '新闻', '35': '漫画', '36': '小说', '37': '技术', '38': '教辅', '39': '问答交流', '40': '搞笑', '41': '杂志', '42': '百科', '43': '影视娱乐', '44': '求职', '45': '兼职', '46': '视频', '47': '短视频', '48': '音乐', '49': '直播', '50': '电台', '51': 'K歌', '52': '成人', '53': '中小学', '54': '职考', '55': '公务员', '56': '英语', '57': '视频教育', '58': '高等教育', '59': '成人教育', '60': '艺术', '61': '语言(非英语)', '62': '旅游资讯', '63': '综合预定', '64': '民航', '65': '铁路', '66': '酒店', '67': '行程管理', '68': '民宿短租', '69': '出国', '70': '工具', '71': '亲子儿童', '72': '母婴', '73': '驾校', '74': '违章', '75': '汽车咨询', '76': '汽车交易', '77': '日常养车', '78': '行车辅助', '79': '租房', '80': '买房', '81': '装修家居', '82': '电子产品', '83': '问诊挂号', '84': '养生保健', '85': '医疗服务', '86': '减肥瘦身', '87': '美妆美业', '88': '菜谱', '89': '餐饮店', '90': '体育咨讯', '91': '运动健身', '92': '支付', '93': '保险', '94': '股票', '95': '借贷', '96': '理财', '97': '彩票', '98': '记账', '99': '银行', '100': '美颜', '101': '影像剪辑', '102': '摄影修图', '103': '相机', '104': '绘画', '105': '二手', '106': '电商', '107': '团购', '108': '外卖', '109': '电影票务', '110': '社区超市', '111': '购物咨询', '112': '笔记', '113': '办公', '114': '日程管理', '115': '女性', '116': '经营', '117': '收款', '118': '其他'}

        dataset = self.load_dataset(filename)
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
                    prompt = f'这是关于{l}的应用程序:{content}'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": label, "candidates": candidates})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class TNEWSDataset(Dataset):
    def __init__(self, args, filename, tokenizer):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'100': '故事',
                           '101': '文化',
                           '102': '娱乐',
                           '103': '体育',
                           '104': '财经',
                           '106': '房产',
                           '107': '汽车',
                           '108': '教育',
                           '109': '科技',
                           '110': '军事',
                           '112': '旅游',
                           '113': '世界',
                           '114': '股票',
                           '115': '农业',
                           '116': '游戏'}

        dataset = self.load_dataset(filename)
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
                    prompt = f'这是关于{l}的文章:{content}'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": label, "candidates": candidates})

        logger.info(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets