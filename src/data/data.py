import os
import json
import re
import random
from typing import Tuple, List

import torch
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from src.utils import logger, RESOURCE_PATH
from src.utils.modeling_utils import _prepare_decoder_attention_mask, qwen_make_context
from src.utils.file_utils import print_rank_0


def chatglm3_encode(tokenizer: PreTrainedTokenizerBase,
                    query: str,
                    label: str = None,
                    system: str = "",
                    max_length: int = 1024,
                    is_prefix: bool = True
                    ) -> Tuple[List[int], List[int], List[int]]:
    '''Use chatglm3 tokenizer to encode prompt + label with "longest_first" truncation strategy

    :param tokenizer:
    :param prompt:
    :param label:
    :param system:
    :param max_length:
    :return:
    '''
    prefix_tokens = tokenizer.get_prefix_tokens()
    role_tokens_1 = [tokenizer.get_command(f"<|user|>")] + tokenizer.encode(f"\n", add_special_tokens=False)
    # Process `system` and `query`
    if is_prefix:
        system_ids = tokenizer.encode(system + "\n\n", add_special_tokens=False) if len(system) > 0 else []
        query_ids = tokenizer.encode(" " + query, add_special_tokens=False)[1:]
    else:
        system_ids = tokenizer.encode(" \n\n" + system, add_special_tokens=False)[1:] if len(system) > 0 else []
        query_ids = tokenizer.encode(query, add_special_tokens=False)
    # Process `label`
    role_tokens_2 = [tokenizer.get_command(f"<|assistant|>")]
    if label is not None:
        label_ids = tokenizer.encode(label, add_special_tokens=False)
        end_tokens = [tokenizer.get_command("<eos>")]
    else:
        label_ids = []
        end_tokens = []
    # Remove overflowing tokens
    num_tokens_to_remove = len(prefix_tokens) + len(role_tokens_1) + len(query_ids) + len(system_ids) + \
                           len(role_tokens_2) + len(label_ids) + len(end_tokens) - max_length
    if num_tokens_to_remove > 0:
        for _ in range(num_tokens_to_remove):
            if len(query_ids) + len(system_ids) > len(label_ids) and len(query_ids) > 0:
                query_ids.pop()
            elif len(label_ids) > 0:
                label_ids.pop()
            else:
                logger.warn("removing system tokens due to tokens overflowing")
                system_ids.pop()
        if label is not None:
            label_ids += end_tokens
    else:
        if label is not None:
            label_ids += end_tokens
        label_ids += [tokenizer.pad_token_id] * -num_tokens_to_remove

    if is_prefix:
        prompt_ids = prefix_tokens + role_tokens_1 + system_ids + query_ids + role_tokens_2
    else:
        prompt_ids = prefix_tokens + role_tokens_1 + query_ids + system_ids + role_tokens_2
    input_ids = prompt_ids + label_ids
    labels = [tokenizer.pad_token_id] * len(prompt_ids) + label_ids
    assert len(input_ids) == len(labels) == max_length
    return input_ids, labels, prompt_ids


def chatglm2_encode(tokenizer: PreTrainedTokenizerBase,
                    query: str,
                    label: str = None,
                    system: str = "",
                    max_length: int = 1024,
                    is_prefix: bool = True
                    ) -> Tuple[List[int], List[int], List[int]]:
    '''Use chatglm2 tokenizer to encode prompt + label with "longest_first" truncation strategy

    :param tokenizer:
    :param prompt:
    :param label:
    :param system:
    :param max_length:
    :return:
    '''
    gmask_id = tokenizer.get_command("[gMASK]")
    sop_id = tokenizer.get_command("sop")
    eop_id = tokenizer.get_command("eop")
    # [Round {1}]\n\n问：
    ids1 = [790, 30951, 517, 30910, 30939, 30996, 13, 13, 54761, 31211]
    # \n\n答：
    ids2 = [13, 13, 55437, 31211]
    if len(system) > 0:
        if is_prefix:
            system_ids = tokenizer.encode(" " + system + "\n\n", add_special_tokens=False)[1:]
        else:
            system_ids = tokenizer.encode(" \n\n" + system, add_special_tokens=False)[1:]
    else:
        system_ids = []
    query_ids = tokenizer.encode(" " + query, add_special_tokens=False)[1:]
    if label is not None:
        label_ids = tokenizer.encode(label, add_special_tokens=False)
        num_special_tokens = 3
    else:
        label_ids = []
        num_special_tokens = 2
    num_tokens_to_remove = len(ids1) + len(query_ids) + len(system_ids) + len(ids2) + \
                           len(label_ids) + num_special_tokens - max_length
    if num_tokens_to_remove > 0:
        for _ in range(num_tokens_to_remove):
            if len(query_ids) + len(system_ids) > len(label_ids) and len(query_ids) > 0:
                query_ids.pop()
            elif len(label_ids) > 0:
                label_ids.pop()
            else:
                logger.warn("removing system tokens due to tokens overflowing")
                system_ids.pop()
        if label is not None:
            label_ids += [eop_id]
    else:
        if label is not None:
            label_ids += [eop_id]
        label_ids += [tokenizer.pad_token_id] * -num_tokens_to_remove
    if is_prefix:
        prompt_ids = [gmask_id, sop_id] + ids1 + system_ids + query_ids + ids2
    else:
        prompt_ids = [gmask_id, sop_id] + ids1 + query_ids + system_ids + ids2
    input_ids = prompt_ids + label_ids
    labels = [tokenizer.pad_token_id] * len(prompt_ids) + label_ids
    assert len(input_ids) == len(labels) == max_length
    return input_ids, labels, prompt_ids


class DataCollatorReward:
    def __call__(self, data):
        has_attention_mask = 'attention_mask' in data[0]
        batch = {
            "chosen_input_ids": torch.stack([f['input_ids'] for f in data]),
            "chosen_attention_mask": torch.stack([f['attention_mask'] for f in data]) if has_attention_mask else None,
            # "input_ids": torch.cat([f[0] for f in data] + [f[2] for f in data]),
            # "attention_mask": torch.cat([f[1] for f in data] + [f[3] for f in data]),
            # "labels": torch.tensor([0] * len(data) + [1] * len(data))
        }
        return batch


class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
        batch = {}
        pad_token_id = data[-1][-1]

        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)

        ### make sure the final ouput is a seqence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompt"] = F.pad(prompt,
                                    pad=(pad_length, 0),
                                    mode='constant',
                                    value=pad_token_id)
            batch["prompt_att_mask"] = F.pad(prompt_mask,
                                             pad=(pad_length, 0),
                                             mode='constant',
                                             value=0)
        else:
            batch["prompt"] = prompt
            batch["prompt_att_mask"] = prompt_mask
        batch["prompt"] = batch["prompt"].flip(1)
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        return batch


class PretrainDataset(Dataset):
    def __init__(self, args, filename, tokenizer, concat_samples=True):
        self.args = args
        self.tokenizer = tokenizer
        self.concat_samples = concat_samples
        self.model_name_or_path = args.model_name_or_path if hasattr(args,
                                                                     "model_name_or_path") else args.actor_model_path

        self.post_list = self.load_dataset(filename)
        for k in range(5):
            print_rank_0(f"PretrainDataset sample-{k}\n: {self.post_list[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        data = self.post_list[idx]
        if not self.concat_samples:
            prompt = data['prompt']
            label = data.get('label', None)
            if "glm" in self.model_name_or_path.lower() and "chatglm" not in self.model_name_or_path.lower():
                encoded_prompt = self.tokenizer(prompt, self.tokenizer.mask_token)
                prompt_length = len(encoded_prompt['input_ids'])
                label_length = len(self.tokenizer.tokenize(label)) + 1
                if prompt_length + label_length > self.args.max_length:
                    num_tokens_to_remove = prompt_length + label_length - self.args.max_length
                    for _ in range(num_tokens_to_remove):
                        if prompt_length > label_length:
                            prompt_length -= 1
                        else:
                            label_length -= 1
                else:
                    label_length = self.args.max_length - prompt_length
                assert prompt_length > 0
                assert label_length > 0
                assert prompt_length + label_length == self.args.max_length
                encoded_dict = self.tokenizer(prompt, self.tokenizer.mask_token,
                                              max_length=prompt_length,
                                              truncation="only_first",
                                              return_tensors="pt",
                                              return_attention_mask=True,
                                              return_token_type_ids=False)
                encoded_dict = self.tokenizer.build_inputs_for_generation(encoded_dict, targets=label,
                                                                          max_gen_length=label_length, padding=True)
                return {
                    "input_ids": encoded_dict['input_ids'][0],
                    "position_ids": encoded_dict['position_ids'][0],
                    "attention_mask": encoded_dict['attention_mask'][0],
                    "labels": encoded_dict['labels'][0],
                }
            else:
                if "chatglm2" in self.model_name_or_path.lower():
                    prompt = f"[Round {1}]\n\n问：{prompt}\n\n答："
                    label = label
                elif "chatglm" in self.model_name_or_path.lower():
                    prompt = f"[Round {0}]\n问：{prompt}\n答："
                    label = label
                elif "vicuna" in self.model_name_or_path.lower():
                    prompt += "\n\n" + label
                    label = None
                else:
                    label = None
                encoded_dict = self.tokenizer(prompt, label,
                                              max_length=self.args.max_length,
                                              truncation="longest_first",
                                              padding="max_length",
                                              return_token_type_ids=False,
                                              return_tensors="pt", )
                if "pangu" in self.model_name_or_path.lower():
                    return {
                        "input_ids": encoded_dict['input_ids'],
                        "attention_mask": encoded_dict['attention_mask'],
                        "labels": encoded_dict['input_ids'],
                    }
                else:
                    result = {
                        "input_ids": encoded_dict['input_ids'][0],
                        "labels": encoded_dict['input_ids'][0],
                    }
                    if 'attention_mask' in encoded_dict:
                        result["attention_mask"] = encoded_dict['attention_mask'][0]
                    return result
        else:
            eos_ids = data['eos_ids']
            input_ids = data['input_ids']
            combined_attention_mask = torch.full((self.args.max_length, self.args.max_length),
                                                 torch.tensor(torch.finfo(torch.float16).min))
            for i in range(len(eos_ids) - 1):
                attention_mask = torch.ones((1, eos_ids[i + 1] - eos_ids[i]), dtype=torch.long)
                attention_mask = _prepare_decoder_attention_mask(attention_mask, attention_mask.shape,
                                                                 input_embeds=torch.ones(1, dtype=torch.float16,
                                                                                         device="cpu"),
                                                                 past_key_values_length=0)
                logger.debug(f"{i}-th sample, shape: {attention_mask.shape}, attention_mask: {attention_mask}")
                combined_attention_mask[eos_ids[i]:eos_ids[i + 1], eos_ids[i]:eos_ids[i + 1]] = attention_mask
            logger.debug(f"shape: {combined_attention_mask.shape}, combined_attention_mask: {combined_attention_mask}")
            if "chatglm2" in self.model_name_or_path.lower():
                return {
                    "input_ids": input_ids,
                    "labels": input_ids,
                    "full_attention_mask": combined_attention_mask,
                }
            else:
                return {
                    "input_ids": input_ids,
                    "labels": input_ids,
                    "attention_mask": combined_attention_mask,
                }

    def load_dataset(self, filename):
        discard = 0
        datasets = []
        with open(filename, "r", encoding="utf-8") as f:
            data = []
            eos_ids = [0]
            length = 0
            for i, line in tqdm(enumerate(f), desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                prompt = str(item['prompt'])
                label = item.get('label', None)
                if len(prompt) <= 0:
                    discard += 1
                    continue
                if not self.concat_samples:
                    datasets.append({"prompt": prompt, "label": label})
                else:
                    if "chatglm2" not in self.model_name_or_path.lower():
                        assert "glm" not in self.model_name_or_path.lower(), \
                            "Concatenating samples for GLM or ChatGLM not implemented yet"
                    if "chatglm2" in self.model_name_or_path.lower():
                        prompt = f"[Round {1}]\n\n问：{prompt}\n\n答："
                    else:
                        prompt = prompt if label is None else "\n\n".join((prompt, label))
                        label = None
                    token_ids = self.tokenizer.encode(prompt, label,
                                                      max_length=self.args.max_length - length,
                                                      truncation="longest_first")
                    if length + len(token_ids) < self.args.max_length:
                        data.extend(token_ids)
                        length += len(token_ids)
                        eos_ids.append(length)
                    else:
                        data.extend(token_ids[:(self.args.max_length - length)])
                        eos_ids.append(self.args.max_length)
                        datasets.append({"input_ids": data, "eos_ids": eos_ids})
                        data = []
                        eos_ids = [0]
                        length = 0
        print_rank_0(
            f"Finished loading {os.path.basename(filename)}, # samples: {len(datasets)}, # discarded: {discard}")

        return datasets


class SFTDataset(Dataset):
    def __init__(self, args, filename, tokenizer, concat_samples=True):
        self.args = args
        self.tokenizer = tokenizer
        self.concat_samples = concat_samples
        self.model_name_or_path = args.model_name_or_path if hasattr(args,
                                                                     "model_name_or_path") else args.actor_model_path

        self.post_list = self.load_dataset(filename)
        for k in range(5):
            print_rank_0(f"SFTDataset sample-{k}\n: {self.post_list[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        data = self.post_list[idx]
        if not self.concat_samples:
            prompt = data['prompt']
            label = data['label']
            prefix = data['prefix']
            system = data['system']
            if "glm" in self.model_name_or_path.lower() and "chatglm" not in self.model_name_or_path.lower():
                encoded_prompt = self.tokenizer(prompt, prefix + self.tokenizer.mask_token)
                prompt_length = len(encoded_prompt['input_ids'])
                label_length = len(self.tokenizer.tokenize(label)) + 1
                if prompt_length + label_length > self.args.max_length:
                    num_tokens_to_remove = prompt_length + label_length - self.args.max_length
                    for _ in range(num_tokens_to_remove):
                        if prompt_length > label_length:
                            prompt_length -= 1
                        else:
                            label_length -= 1
                else:
                    label_length = self.args.max_length - prompt_length
                assert prompt_length > 0
                assert label_length > 0
                assert prompt_length + label_length == self.args.max_length
                encoded_dict = self.tokenizer(prompt, prefix + self.tokenizer.mask_token,
                                              max_length=prompt_length,
                                              truncation="only_first",
                                              return_tensors="pt",
                                              return_attention_mask=True,
                                              return_token_type_ids=False)
                encoded_dict = self.tokenizer.build_inputs_for_generation(encoded_dict, targets=label,
                                                                          max_gen_length=label_length, padding=True)
                return {
                    "input_ids": encoded_dict['input_ids'][0],
                    "position_ids": encoded_dict['position_ids'][0],
                    "attention_mask": encoded_dict['attention_mask'][0],
                    "labels": encoded_dict['labels'][0],
                }
            elif "pangu" in self.model_name_or_path.lower():
                label = prefix + label
                encoded_dict = self.tokenizer(prompt, label,
                                              max_length=self.args.max_length,
                                              truncation="longest_first",
                                              padding="max_length",
                                              return_token_type_ids=False,
                                              return_tensors="pt", )
                return {
                    "input_ids": encoded_dict['input_ids'],
                    "attention_mask": encoded_dict['attention_mask'],
                    "labels": encoded_dict['input_ids'],
                }
            elif "chatglm3" in self.model_name_or_path.lower():
                input_ids, labels, _ = chatglm3_encode(self.tokenizer, prompt, label, system, self.args.max_length)
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    # "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            elif "chatglm2" in self.model_name_or_path.lower():
                input_ids, labels, _ = chatglm2_encode(self.tokenizer, prompt, label, system, self.args.max_length)
                # gmask_id = self.tokenizer.get_command("[gMASK]")
                # sop_id = self.tokenizer.get_command("sop")
                # eop_id = self.tokenizer.get_command("eop")
                # # [Round {1}]\n\n问：
                # ids1 = [790, 30951, 517, 30910, 30939, 30996, 13, 13, 54761, 31211]
                # # \n\n答：
                # ids2 = [13, 13, 55437, 31211]
                # prompt = "\n\n".join((system, prompt))
                # prompt_ids = self.tokenizer.encode(" " + prompt, add_special_tokens=False)[1:]
                # label_ids = self.tokenizer.encode(label, add_special_tokens=False)
                # num_tokens_to_remove = len(ids1) + len(prompt_ids) + len(ids2) + len(label_ids) + 3 - self.args.max_length
                # if num_tokens_to_remove > 0:
                #     for _ in range(num_tokens_to_remove):
                #         if len(prompt_ids) > len(label_ids):
                #             prompt_ids.pop()
                #         else:
                #             label_ids.pop()
                #     prompt_ids = [gmask_id, sop_id] + ids1 + prompt_ids + ids2
                #     label_ids = label_ids + [eop_id]
                # else:
                #     prompt_ids = [gmask_id, sop_id] + ids1 + prompt_ids + ids2
                #     label_ids = label_ids + [eop_id] + [self.tokenizer.pad_token_id] * -num_tokens_to_remove
                # input_ids = prompt_ids + label_ids
                # labels = [self.tokenizer.pad_token_id] * len(prompt_ids) + label_ids
                # assert len(input_ids) == len(labels) == self.args.max_length
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    # "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            elif "chatglm" in self.model_name_or_path.lower():
                prompt = f"[Round {0}]\n问：{prompt}\n答："
                encoded_dict = self.tokenizer(prompt, label,
                                              max_length=self.args.max_length,
                                              truncation="longest_first",
                                              padding="max_length",
                                              return_token_type_ids=False,
                                              return_tensors="pt", )
                return {
                    "input_ids": encoded_dict['input_ids'][0],
                    "attention_mask": encoded_dict['attention_mask'][0],
                    "labels": encoded_dict['input_ids'][0],
                }
            else:
                encoded_dict = self.tokenizer(prompt, label,
                                              max_length=self.args.max_length,
                                              truncation="longest_first",
                                              padding="max_length",
                                              return_token_type_ids=False,
                                              return_tensors="pt", )
                result = {
                    "input_ids": encoded_dict['input_ids'][0],
                    "labels": encoded_dict['input_ids'][0],
                }
                if 'attention_mask' in encoded_dict:
                    result["attention_mask"] = encoded_dict['attention_mask'][0]
                return result
        else:
            eos_ids = data['eos_ids']
            input_ids = data['input_ids']
            combined_attention_mask = torch.full((self.args.max_length, self.args.max_length),
                                                 torch.tensor(torch.finfo(torch.float16).min))
            for i in range(len(eos_ids) - 1):
                attention_mask = torch.ones((1, eos_ids[i + 1] - eos_ids[i]), dtype=torch.long)
                attention_mask = _prepare_decoder_attention_mask(attention_mask, attention_mask.shape,
                                                                 input_embeds=torch.ones(1, dtype=torch.float16,
                                                                                         device="cpu"),
                                                                 past_key_values_length=0)
                logger.debug(f"{i}-th sample, shape: {attention_mask.shape}, attention_mask: {attention_mask}")
                combined_attention_mask[eos_ids[i]:eos_ids[i + 1], eos_ids[i]:eos_ids[i + 1]] = attention_mask
            logger.debug(f"shape: {combined_attention_mask.shape}, combined_attention_mask: {combined_attention_mask}")
            if "chatglm2" in self.model_name_or_path.lower():
                return {
                    "input_ids": input_ids,
                    "labels": input_ids,
                    "full_attention_mask": combined_attention_mask,
                }
            else:
                return {
                    "input_ids": input_ids,
                    "labels": input_ids,
                    "attention_mask": combined_attention_mask,
                }

    def load_dataset(self, filename):
        discard = 0
        datasets = []
        with open(filename, "r", encoding="utf-8") as f:
            data = []
            eos_ids = [0]
            length = 0
            for i, line in tqdm(enumerate(f), desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                data_type = item.get('data_type', "human_generated")
                if data_type != "human_generated":
                    continue
                prompt = str(item['prompt'])
                label = str(item['answers'][0]['answer'])
                score = item['answers'][0]['score']
                prefix = item.get('prefix', "")
                system = item.get('system', "")
                if len(prompt) <= 0 or len(label) <= 0:
                    discard += 1
                    continue

                if not self.concat_samples:
                    datasets.append({"prompt": prompt, "label": label, "prefix": prefix, "system": system})
                else:
                    if "chatglm2" not in self.model_name_or_path.lower():
                        assert "glm" not in self.model_name_or_path.lower(), \
                            "Concatenating samples for GLM or ChatGLM not implemented yet"
                    else:
                        if "chatglm2" in self.model_name_or_path.lower():
                            prompt = f"[Round {1}]\n\n问：{prompt}\n\n答："
                        else:
                            prompt = prompt if label is None else "\n\n".join((prompt, label))
                            label = None
                        token_ids = self.tokenizer.encode(prompt, label,
                                                          max_length=self.args.max_length - length,
                                                          truncation="longest_first")
                        if length + len(token_ids) < self.args.max_length:
                            data.extend(token_ids)
                            length += len(token_ids)
                            eos_ids.append(length)
                        else:
                            data.extend(token_ids[:(self.args.max_length - length)])
                            eos_ids.append(self.args.max_length)
                            datasets.append({"input_ids": data, "eos_ids": eos_ids})
                            data = []
                            eos_ids = [0]
                            length = 0

        print_rank_0(
            f"Finished loading {os.path.basename(filename)}, # samples: {len(datasets)}, # discarded: {discard}")

        return datasets


class PairwiseDataset(Dataset):
    def __init__(self, args, filename, tokenizer):
        self.pairs = self.load_dataset(filename)
        self.args = args
        self.tokenizer = tokenizer

        for k in range(5):
            print_rank_0(f"PairwiseDataset sample-{k}\n: {self.pairs[k]}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        prompt = pair["prompt"]
        chosen_answer = pair["chosen_answer"]
        rejected_answer = pair["rejected_answer"]
        prefix = pair['prefix']
        system = pair['system']
        if "pangu" in self.args.model_name_or_path.lower():
            chosen_encodings_dict = self.tokenizer(prompt, prefix + chosen_answer, max_length=self.args.max_length,
                                                   truncation="longest_first", padding="max_length",
                                                   return_tensors="pt",
                                                   return_token_type_ids=False)
            rejected_encodings_dict = self.tokenizer(prompt, prefix + rejected_answer, max_length=self.args.max_length,
                                                     truncation="longest_first", padding="max_length",
                                                     return_tensors="pt",
                                                     return_token_type_ids=False)
            return {
                "chosen_input_ids": chosen_encodings_dict["input_ids"],
                "chosen_attention_mask": chosen_encodings_dict["attention_mask"],
                "rejected_input_ids": rejected_encodings_dict["input_ids"],
                "rejected_attention_mask": rejected_encodings_dict["attention_mask"],
                "labels": rejected_encodings_dict["input_ids"],
            }
        elif "chatglm3" in self.args.model_name_or_path.lower():
            chosen_input_ids, labels, _ = chatglm3_encode(self.tokenizer, prompt, chosen_answer, system,
                                                          self.args.max_length)
            rejected_input_ids, labels, _ = chatglm3_encode(self.tokenizer, prompt, rejected_answer, system,
                                                            self.args.max_length)
            return {
                "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
                "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }
        elif "chatglm2" in self.args.model_name_or_path.lower():
            chosen_input_ids, labels, _ = chatglm2_encode(self.tokenizer, prompt, chosen_answer, system,
                                                          self.args.max_length)
            rejected_input_ids, labels, _ = chatglm2_encode(self.tokenizer, prompt, rejected_answer, system,
                                                            self.args.max_length)
            return {
                "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
                "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }
        elif "chatglm" in self.args.model_name_or_path.lower():
            prompt = f"[Round {0}]\n问：{prompt}\n答："
            chosen_encodings_dict = self.tokenizer(prompt, chosen_answer, max_length=self.args.max_length,
                                                   truncation="longest_first", padding="max_length",
                                                   return_tensors="pt")
            rejected_encodings_dict = self.tokenizer(prompt, rejected_answer, max_length=self.args.max_length,
                                                     truncation="longest_first", padding="max_length",
                                                     return_tensors="pt")
            return {
                "chosen_input_ids": chosen_encodings_dict["input_ids"][0],
                "rejected_input_ids": rejected_encodings_dict["input_ids"][0],
                "labels": rejected_encodings_dict["input_ids"][0],
            }
        elif "glm" in self.args.model_name_or_path.lower():
            chosen_prompt_length = len(self.tokenizer.tokenize(prompt + prefix)) + 4
            rejected_prompt_length = chosen_prompt_length
            chosen_answer_length = len(self.tokenizer.tokenize(chosen_answer)) + 1
            if chosen_prompt_length + chosen_answer_length > self.args.max_length:
                if chosen_prompt_length >= chosen_answer_length:
                    chosen_prompt_length -= chosen_prompt_length + chosen_answer_length - self.args.max_length
                else:
                    chosen_answer_length -= chosen_prompt_length + chosen_answer_length - self.args.max_length
            else:
                chosen_answer_length = self.args.max_length - chosen_prompt_length
            chosen_encoded_dict = self.tokenizer(prompt, prefix + self.tokenizer.mask_token,
                                                 max_length=chosen_prompt_length,
                                                 truncation="only_first",
                                                 return_tensors="pt",
                                                 return_token_type_ids=False)
            chosen_encodings_dict = self.tokenizer.build_inputs_for_generation(chosen_encoded_dict,
                                                                               targets=chosen_answer,
                                                                               max_gen_length=chosen_answer_length,
                                                                               padding=True)

            rejected_answer_length = len(self.tokenizer.tokenize(rejected_answer)) + 1
            if rejected_prompt_length + rejected_answer_length > self.args.max_length:
                if rejected_prompt_length >= rejected_answer_length:
                    rejected_prompt_length -= rejected_prompt_length + rejected_answer_length - self.args.max_length
                else:
                    rejected_answer_length -= rejected_prompt_length + rejected_answer_length - self.args.max_length
            else:
                rejected_answer_length = self.args.max_length - rejected_prompt_length
            rejected_encoded_dict = self.tokenizer(prompt, prefix + self.tokenizer.mask_token,
                                                   max_length=rejected_prompt_length,
                                                   truncation="only_first",
                                                   return_tensors="pt",
                                                   return_token_type_ids=False)
            rejected_encodings_dict = self.tokenizer.build_inputs_for_generation(rejected_encoded_dict,
                                                                                 targets=rejected_answer,
                                                                                 max_gen_length=rejected_answer_length,
                                                                                 padding=True)
            return {
                "chosen_input_ids": chosen_encodings_dict["input_ids"][0],
                "chosen_attention_mask": chosen_encodings_dict["attention_mask"][0],
                "chosen_position_ids": chosen_encodings_dict["position_ids"][0],
                "rejected_input_ids": rejected_encodings_dict["input_ids"][0],
                "rejected_attention_mask": rejected_encodings_dict["attention_mask"][0],
                "rejected_position_ids": rejected_encodings_dict["position_ids"][0],
                "labels": rejected_encodings_dict["input_ids"][0],
            }
        else:
            raise ValueError(f"Unsupported model name: {self.args.model_name_or_path}")

    @staticmethod
    def load_dataset(filename):
        discard = 0
        pairs = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                prompt = str(item['prompt'])
                answers = item['answers']
                prefix = item.get('prefix', "")
                system = item.get('system', "")
                chosen_answer, rejected_answer = None, None
                for i in range(len(answers) - 1):
                    answer_1 = str(answers[i]["answer"])
                    answer_1_score = answers[i]["score"]
                    answer_2 = str(answers[i + 1]["answer"])
                    answer_2_score = answers[i + 1]["score"]
                    if answer_1_score > answer_2_score:
                        chosen_answer = answer_1
                    rejected_answer = answer_2
                    if chosen_answer is not None and rejected_answer is not None \
                            and len(prompt) > 0 and len(chosen_answer) > 0 and len(rejected_answer) > 0 \
                            and chosen_answer != rejected_answer:
                        pair = {
                            "prompt": prompt,
                            "prefix": prefix,
                            "system": system,
                            "chosen_answer": chosen_answer,
                            "rejected_answer": rejected_answer
                        }
                        pairs.append(pair)
                    else:
                        discard += 1

        print_rank_0(f"Finished loading {os.path.basename(filename)}, # pairs: {len(pairs)}, # discarded: {discard}")

        return pairs


class RLHFDataset(Dataset):
    def __init__(self, args, filename, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        assert tokenizer.padding_side == "left", "In RLHF training, need to set padding_side to 'left'"

        self.post_list = self.load_dataset(filename)
        for k in range(5):
            print_rank_0(f"RLHFDataset sample-{k}\n: {self.post_list[k]}")

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        data = self.post_list[idx]
        prompt = data['prompt']
        prefix = data['prefix']
        system = data['system']
        if "pangu" in self.args.actor_model_path:
            encoded_dict = self.tokenizer(prompt, self.tokenizer.sep_token + prefix,
                                          max_length=self.args.max_prompt_length,
                                          # padding="max_length",
                                          truncation="only_first", add_special_tokens=False,
                                          return_tensors="pt", return_token_type_ids=False)
            return {
                "input_ids": encoded_dict['input_ids'][0],
                "attention_mask": encoded_dict['attention_mask'][0],
                # "labels": encoded_dict['input_ids'],
            }
        elif "chatglm" in self.args.actor_model_path:
            prompt = "\n\n".join((system, prompt))
            prompt = f"[Round {1}]\n\n问：{prompt}\n\n答：" if "chatglm2" in self.args.actor_model_path else f"[Round {0}]\n问：{prompt}\n答："
            encoded_dict = self.tokenizer(prompt, max_length=self.args.max_prompt_length,
                                          return_tensors="pt", truncation="only_first")
            return {
                "input_ids": encoded_dict['input_ids'][0],
            }
        elif "glm" in self.args.actor_model_path:
            # encoded_prompt = self.tokenizer(prompt, prefix + self.tokenizer.mask_token)
            # prompt_length = len(encoded_prompt['input_ids'])
            encoded_dict = self.tokenizer(prompt, prefix + self.tokenizer.mask_token,
                                          max_length=self.args.max_prompt_length,
                                          # padding="max_length",
                                          truncation="only_first",
                                          return_tensors="pt",
                                          return_token_type_ids=False)
            encoded_dict = self.tokenizer.build_inputs_for_generation(encoded_dict,
                                                                      max_gen_length=self.args.max_gen_length,
                                                                      padding=True)

            return {
                "input_ids": encoded_dict['input_ids'][0],
                "position_ids": encoded_dict['position_ids'][0],
                "generation_attention_mask": encoded_dict['generation_attention_mask'][0],
                # "labels": encoded_dict['labels'][0],
            }
        else:
            raise ValueError(f"Unsupported model name: {self.args.model_name_or_path}")

    @staticmethod
    def load_dataset(filename):
        discard = 0
        datasets = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f), desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                data_type = item.get('data_type', "human_generated")
                if data_type != "human_generated":
                    continue
                prompt = str(item['prompt'])
                prefix = item.get('prefix', "")
                system = item.get('system', "")

                if len(prompt) <= 0:
                    discard += 1
                    continue
                datasets.append({"prompt": prompt, "system": system, "prefix": prefix})
        print_rank_0(
            f"Finished loading {os.path.basename(filename)}, # samples: {len(datasets)}, # discarded: {discard}")

        return datasets


class PPODataset:
    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def separate(self):
        small_dataset = []
        for large_batch in self.dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size] if v is not None else None
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i + self.small_batch_size])
        self.free()

        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.separate()
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        self.dataset = []


class DPODataset(Dataset):
    def __init__(self, args, filename, tokenizer):
        self.pairs = self.load_dataset(filename)
        self.args = args
        self.tokenizer = tokenizer

        for k in range(5):
            print_rank_0(f"DPODataset sample-{k}\n: {self.pairs[k]}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        index = pair["index"]
        prompt = pair["prompt"]
        chosen_answer = pair["chosen_answer"]
        rejected_answer = pair["rejected_answer"]
        prefix = pair['prefix']
        system = pair['system']
        if "pangu" in self.args.model_name_or_path.lower():
            chosen_encodings_dict = self.tokenizer(prompt, prefix + chosen_answer, max_length=self.args.max_length,
                                                   truncation="longest_first", padding="max_length",
                                                   return_tensors="pt",
                                                   return_token_type_ids=False)
            rejected_encodings_dict = self.tokenizer(prompt, prefix + rejected_answer, max_length=self.args.max_length,
                                                     truncation="longest_first", padding="max_length",
                                                     return_tensors="pt",
                                                     return_token_type_ids=False)
            return {
                "chosen_input_ids": chosen_encodings_dict["input_ids"],
                "chosen_attention_mask": chosen_encodings_dict["attention_mask"],
                "rejected_input_ids": rejected_encodings_dict["input_ids"],
                "rejected_attention_mask": rejected_encodings_dict["attention_mask"],
                "labels": rejected_encodings_dict["input_ids"],
            }
        elif "chatglm3" in self.args.model_name_or_path.lower():
            chosen_input_ids, chosen_labels, _ = chatglm3_encode(self.tokenizer, prompt, chosen_answer, system,
                                                                 self.args.max_length)
            rejected_input_ids, rejected_labels, _ = chatglm3_encode(self.tokenizer, prompt, rejected_answer, system,
                                                                     self.args.max_length)
            return {
                "index": torch.tensor(index, dtype=torch.long),
                "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
                "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
                "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
                "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long)
            }
        elif "chatglm2" in self.args.model_name_or_path.lower():
            chosen_input_ids, chosen_labels, _ = chatglm2_encode(self.tokenizer, prompt, chosen_answer, system,
                                                                 self.args.max_length)
            rejected_input_ids, rejected_labels, _ = chatglm2_encode(self.tokenizer, prompt, rejected_answer, system,
                                                                     self.args.max_length)
            return {
                "index": torch.tensor(index, dtype=torch.long),
                "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
                "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
                "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
                "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long)
            }
        elif "chatglm" in self.args.model_name_or_path.lower():
            prompt = f"[Round {0}]\n问：{prompt}\n答："
            chosen_encodings_dict = self.tokenizer(prompt, chosen_answer, max_length=self.args.max_length,
                                                   truncation="longest_first", padding="max_length",
                                                   return_tensors="pt")
            rejected_encodings_dict = self.tokenizer(prompt, rejected_answer, max_length=self.args.max_length,
                                                     truncation="longest_first", padding="max_length",
                                                     return_tensors="pt")
            return {
                "chosen_input_ids": chosen_encodings_dict["input_ids"][0],
                "rejected_input_ids": rejected_encodings_dict["input_ids"][0],
                "labels": rejected_encodings_dict["input_ids"][0],
            }
        else:
            raise ValueError(f"Unsupported model name: {self.args.model_name_or_path}")

    @staticmethod
    def load_dataset(filename):
        discard = 0
        index = 1
        pairs = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {os.path.basename(filename)}"):
                item = json.loads(line)
                prompt = str(item['prompt'])
                answers = item['answers']
                prefix = item.get('prefix', "")
                system = item.get('system', "")
                chosen_answer, rejected_answer = None, None
                for i in range(len(answers) - 1):
                    answer_1 = str(answers[i]["answer"])
                    answer_1_score = answers[i]["score"]
                    answer_2 = str(answers[i + 1]["answer"])
                    answer_2_score = answers[i + 1]["score"]
                    if answer_1_score > answer_2_score:
                        chosen_answer = answer_1
                    rejected_answer = answer_2
                    if chosen_answer is not None and rejected_answer is not None \
                            and len(prompt) > 0 and len(chosen_answer) > 0 and len(rejected_answer) > 0 \
                            and chosen_answer != rejected_answer:
                        pair = {
                            "index": index,
                            "prompt": prompt,
                            "prefix": prefix,
                            "system": system,
                            "chosen_answer": chosen_answer,
                            "rejected_answer": rejected_answer
                        }
                        index += 1
                        pairs.append(pair)
                    else:
                        discard += 1

        print_rank_0(f"Finished loading {os.path.basename(filename)}, # pairs: {len(pairs)}, # discarded: {discard}")

        return pairs


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
            print_rank_0(f"OCNLIDataset sample-{k}\n: {dataset[k]}")

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
                exmample_tokens = self.tokenizer.tokenize(example_prompt + "\n")
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
                # 标注结果有冲突，则忽略
                if label == "-":
                    continue
                for l in self.label_dict.values():
                    prompt = f'{s1}?{l}，{s2}'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": self.label_dict[label]})

        print_rank_0(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

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
            print_rank_0(f"CMNLIDataset sample-{k}\n: {dataset[k]}")

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
                exmample_tokens = self.tokenizer.tokenize(example_prompt + "\n")
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
                # 标注结果有冲突，则忽略
                if label == "-":
                    continue
                for l in self.label_dict.values():
                    prompt = f'{s1}?{l}，{s2}'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": self.label_dict[label]})

        print_rank_0(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

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
            print_rank_0(f"CHIDDataset sample-{k}\n: {dataset[k]}")

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
                exmample_tokens = self.tokenizer.tokenize(example_prompt + "\n")
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

        print_rank_0(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets

    def load_idiom_dict(self):
        idiom_dict = json.load(open(os.path.join(self.args.data_dir, "dev_answer.json"), "r", encoding="utf-8"))
        idiom_dict.update(json.load(open(os.path.join(self.args.data_dir, "train_answer.json"), "r", encoding="utf-8")))

        print_rank_0(f"Finished loading idiom dict")

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
            print_rank_0(f"CMRCDataset sample-{k}\n: {dataset[k]}")

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
                exmample_tokens = self.tokenizer.tokenize(example_prompt + "\n")
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
                    prompt_template = "阅读文章：{context}\n问：{question}\n答："
                    prompt = prompt_template.format(context=context, question=question)
                    if len(prompt) <= 0:
                        continue
                    # if len(prompt) > self.args.max_length:
                    #     idx = len(prompt) - self.args.max_length
                    #     prompt = prompt_template.format(context=context[:-idx], question=question)
                    datasets.append({"prompt": prompt, "label": answers})

        print_rank_0(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

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
            print_rank_0(f"CLUEWSCDataset sample-{k}\n: {dataset[k]}")

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
                exmample_tokens = self.tokenizer.tokenize(example_prompt + "\n")
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
                prompt = text[:span2_index] + span1_text + text[span2_index + len(span2_text):]
                if len(prompt) <= 0:
                    continue
                datasets.append({"prompt": prompt, "label": label})

        print_rank_0(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

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
            print_rank_0(f"C3Dataset sample-{k}\n: {dataset[k]}")

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
                exmample_tokens = self.tokenizer.tokenize(example_prompt + "\n")
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
                    prompt = f"问: {question}\n答:{choice}\n该答案来自对话: {context}"
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": answer, "candidates": choices_padded})

        print_rank_0(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class AFQMCDataset(Dataset):
    def __init__(self, args, eval_filename, tokenizer, train_filename=None):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'0': '不同', '1': '相同'}

        dataset = self.load_dataset(eval_filename)
        if train_filename is not None:
            self.labelled_list = self.load_dataset(eval_filename)
        self.post_list = dataset

        for k in range(5):
            print_rank_0(f"AFQMCDataset sample-{k}\n: {dataset[k]}")

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
                exmample_tokens = self.tokenizer.tokenize(example_prompt + "\n")
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
                    prompt = f'下面两个句子语义{l}:{s1}。{s2}'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": label})

        print_rank_0(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class CSLDataset(Dataset):
    def __init__(self, args, eval_filename, tokenizer, train_filename=None):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'0': '不是', '1': '是'}

        dataset = self.load_dataset(eval_filename)
        if train_filename is not None:
            self.labelled_list = self.load_dataset(eval_filename)
        self.post_list = dataset

        for k in range(5):
            print_rank_0(f"CSLDataset sample-{k}\n: {dataset[k]}")

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
                exmample_tokens = self.tokenizer.tokenize(example_prompt + "\n")
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
                    prompt = f'摘要:{abstract}，关键词:{keyword}{l}真实关键词'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": label})

        print_rank_0(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class IFLYTEKDataset(Dataset):
    def __init__(self, args, eval_filename, tokenizer, train_filename=None):
        self.tokenizer = tokenizer
        self.args = args
        self.label_dict = {'0': '打车', '1': '地图导航', '2': '免费WIFI', '3': '租车', '4': '同城服务', '5': '快递物流',
                           '6': '婚庆', '7': '家政', '8': '公共交通', '9': '政务', '10': '社区服务', '11': '薅羊毛',
                           '12': '魔幻', '13': '仙侠', '14': '卡牌', '15': '飞行空战', '16': '射击游戏',
                           '17': '休闲益智', '18': '动作类', '19': '体育竞技', '20': '棋牌中心', '21': '经营养成',
                           '22': '策略', '23': 'MOBA', '24': '辅助工具', '25': '约会社交', '26': '即时通讯',
                           '27': '工作社交', '28': '论坛圈子', '29': '婚恋社交', '30': '情侣社交', '31': '社交工具',
                           '32': '生活社交', '33': '微博博客', '34': '新闻', '35': '漫画', '36': '小说', '37': '技术',
                           '38': '教辅', '39': '问答交流', '40': '搞笑', '41': '杂志', '42': '百科', '43': '影视娱乐',
                           '44': '求职', '45': '兼职', '46': '视频', '47': '短视频', '48': '音乐', '49': '直播',
                           '50': '电台', '51': 'K歌', '52': '成人', '53': '中小学', '54': '职考', '55': '公务员',
                           '56': '英语', '57': '视频教育', '58': '高等教育', '59': '成人教育', '60': '艺术',
                           '61': '语言(非英语)', '62': '旅游资讯', '63': '综合预定', '64': '民航', '65': '铁路',
                           '66': '酒店', '67': '行程管理', '68': '民宿短租', '69': '出国', '70': '工具',
                           '71': '亲子儿童', '72': '母婴', '73': '驾校', '74': '违章', '75': '汽车咨询',
                           '76': '汽车交易', '77': '日常养车', '78': '行车辅助', '79': '租房', '80': '买房',
                           '81': '装修家居', '82': '电子产品', '83': '问诊挂号', '84': '养生保健', '85': '医疗服务',
                           '86': '减肥瘦身', '87': '美妆美业', '88': '菜谱', '89': '餐饮店', '90': '体育咨讯',
                           '91': '运动健身', '92': '支付', '93': '保险', '94': '股票', '95': '借贷', '96': '理财',
                           '97': '彩票', '98': '记账', '99': '银行', '100': '美颜', '101': '影像剪辑',
                           '102': '摄影修图', '103': '相机', '104': '绘画', '105': '二手', '106': '电商', '107': '团购',
                           '108': '外卖', '109': '电影票务', '110': '社区超市', '111': '购物咨询', '112': '笔记',
                           '113': '办公', '114': '日程管理', '115': '女性', '116': '经营', '117': '收款', '118': '其他'}

        dataset = self.load_dataset(eval_filename)
        if train_filename is not None:
            self.labelled_list = self.load_dataset(eval_filename)
        self.post_list = dataset

        for k in range(5):
            print_rank_0(f"IFLYTEKDataset sample-{k}\n: {dataset[k]}")

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
                exmample_tokens = self.tokenizer.tokenize(example_prompt + "\n")
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
                    prompt = f'这是关于{l}的应用程序:{content}'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": label, "candidates": candidates})

        print_rank_0(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class TNEWSDataset(Dataset):
    def __init__(self, args, eval_filename, tokenizer, train_filename=None):
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

        dataset = self.load_dataset(eval_filename)
        if train_filename is not None:
            self.labelled_list = self.load_dataset(eval_filename)
        self.post_list = dataset

        for k in range(5):
            print_rank_0(f"TNEWSDataset sample-{k}\n: {dataset[k]}")

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
                exmample_tokens = self.tokenizer.tokenize(example_prompt + "\n")
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
                    prompt = f'这是关于{l}的文章:{content}'
                    if len(prompt) <= 0:
                        continue
                    datasets.append({"prompt": prompt, "label": label, "candidates": candidates})

        print_rank_0(f"Finished loading {os.path.basename(filename)}, # discarded: {discard}")

        return datasets


class CEvalDataset(Dataset):
    def __init__(self, args, eval_filename, tokenizer, train_filename=None):
        self.tokenizer = tokenizer
        self.args = args
        self.model_name_or_path = args.model_name_or_path if hasattr(args,
                                                                     "model_name_or_path") else args.actor_model_path
        self.subject_mapping = json.load(open(os.path.join(RESOURCE_PATH, "eval", "ceval", "subject_mapping.json")))
        self.max_length = args.max_length - args.max_length_generation
        self.choices = ["A", "B", "C", "D"]

        self.post_list = self.load_dataset(eval_filename)
        if train_filename is not None:
            self.dev_list = self.load_dataset(train_filename, "dict")

        for k in range(5):
            print_rank_0(f"CEvalDataset sample-{k}\n: {self.post_list[k]}")

    def __len__(self):
        return len(self.post_list)

    def format_example(self, line, include_answer=True, cot=False):
        example = line['question']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        example += '\n答案：'
        if "chatglm" in self.model_name_or_path.lower() or "qwen" in self.model_name_or_path.lower():
            if include_answer:
                if cot:
                    ans = "让我们一步一步思考，\n" + line["explanation"] + f"\n所以答案是{line['answer']}。"
                else:
                    ans = line["answer"]
                m = (example, ans)
                return m
            return example
        else:
            # example = line['question']
            # for choice in self.choices:
            #     example += f'\n{choice}. {line[f"{choice}"]}'
            if include_answer:
                if cot:
                    example += "让我们一步一步思考，\n" + line["explanation"] + f"\n所以答案是{line['answer']}。"
                else:
                    example += line["answer"]
            else:
                if cot:
                    example += "让我们一步一步思考，\n1."
            return example

    def __getitem__(self, idx):
        data = self.post_list[idx]
        subject_name = data['subject_name']
        question = self.format_example(data, include_answer=False, cot=self.args.cot)
        prefix = f"以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。"

        history = []
        if "chatglm" in self.model_name_or_path.lower():
            sep = "\n\n" if "chatglm2" in self.model_name_or_path.lower() else "\n"
            offset = 1 if "chatglm2" in self.model_name_or_path.lower() else 0
            # Few-Shot example construction
            if hasattr(self, "dev_list"):
                history.append(prefix)
                k = self.args.max_few_shot
                dev_list = self.dev_list[subject_name]
                for i in range(min(k, len(dev_list))):
                    prompt, answer = self.format_example(dev_list[i], include_answer=True, cot=self.args.cot)
                    prompt = f"[Round {i + offset}]{sep}问：{prompt}{sep}答：{answer}"
                    history.append(prompt)
            # Concat few-shot/zero-shot examples with question.
            # If length of full prompt exceeds max_length, remove examples until the length is smaller than max_length
            question = f"[Round {len(history) + offset}]{sep}问：{question}{sep}答："
            while True:
                full_prompt = sep.join(history + [question])
                input_ids = self.tokenizer.encode(full_prompt)
                if len(input_ids) <= self.max_length:
                    break
                elif len(history) <= 1:
                    full_prompt = question
                    break
                else:
                    history.pop(-1)
            encoded_dict = self.tokenizer(full_prompt, max_length=self.max_length, return_tensors="pt",
                                          truncation="longest_first")
        elif "qwen" in self.model_name_or_path.lower():
            # Few-Shot example construction
            if hasattr(self, "dev_list"):
                k = self.args.max_few_shot
                dev_list = self.dev_list[subject_name]
                for i in range(min(k, len(dev_list))):
                    history.append(self.format_example(dev_list[i], include_answer=True, cot=self.args.cot))
            full_prompt, input_ids = qwen_make_context(self.tokenizer, question, history, system=prefix,
                                                       max_window_size=self.max_length)
            encoded_dict = {"input_ids": torch.tensor(input_ids, dtype=torch.int64)}
        else:
            # Few-Shot example construction
            if hasattr(self, "dev_list"):
                history.append(prefix)
                k = self.args.max_few_shot
                dev_list = self.dev_list[subject_name]
                for i in range(min(k, len(dev_list))):
                    history.append(self.format_example(dev_list[i], include_answer=True, cot=self.args.cot))
            # Concat few-shot/zero-shot examples with question.
            # If length of full prompt exceeds max_length, remove examples until the length is smaller than max_length
            while True:
                full_prompt = "\n\n".join(history + [question])
                input_ids = self.tokenizer.encode(full_prompt)
                if len(input_ids) <= self.max_length:
                    break
                elif len(history) <= 1:
                    full_prompt = question
                    break
                else:
                    history.pop(-1)
            encoded_dict = self.tokenizer(full_prompt, max_length=self.max_length, return_tensors="pt",
                                          truncation="longest_first")

        logger.debug(f"number of shots: {len(history) - 1}, full prompt: {full_prompt}")

        return {
            "input_ids": encoded_dict["input_ids"],
            "attention_mask": encoded_dict.get("attention_mask", None),
            "number_of_shots": max(len(history) - 1, 0),
            "id": data['id'],
            "subject_name_key": data['subject_name_key'],
            "answer": data.get('answer', None)
        }

    def load_dataset(self, filename, return_format="list"):
        datasets = list() if return_format == "list" else dict()
        dt = os.path.basename(filename)

        for subject_name_key, subject in self.subject_mapping.items():
            subject_name = subject[1]
            if isinstance(datasets, dict):
                datasets[subject_name] = list()
            dev_file_path = os.path.join(filename, f'{subject_name_key}_{dt}.csv')
            dev_df = pd.read_csv(dev_file_path)
            for i, val in dev_df.iterrows():
                d = val.to_dict()
                if isinstance(datasets, dict):
                    datasets[subject_name].append(d)
                else:
                    d['subject_name'] = subject_name
                    d['subject_name_key'] = subject_name_key
                    datasets.append(d)

        print_rank_0(f"Finished loading {dt} dataset")

        return datasets


class MMLUDataset(Dataset):
    def __init__(self, args, eval_filename, tokenizer, train_filename=None):
        self.tokenizer = tokenizer
        self.args = args
        self.model_name_or_path = args.model_name_or_path if hasattr(args,
                                                                     "model_name_or_path") else args.actor_model_path
        self.subject_mapping = json.load(open(os.path.join(RESOURCE_PATH, "eval", "mmlu", "subject_mapping.json")))
        self.choices = ["A", "B", "C", "D"]
        self.max_length = args.max_length - args.max_length_generation

        self.post_list = self.load_dataset(eval_filename)
        if train_filename is not None:
            self.dev_list = self.load_dataset(train_filename, "dict")

        for k in range(5):
            print_rank_0(f"MMLUDataset sample-{k}\n: {self.post_list[k]}")

    def __len__(self):
        return len(self.post_list)

    def format_example(self, line, include_answer=True):
        example = line['question']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        example += '\nAnswer：'
        if "chatglm" in self.model_name_or_path.lower() or "qwen" in self.model_name_or_path.lower():
            if include_answer:
                ans = line["answer"]
                m = (example, ans)
                return m
            return example
        else:
            # example = line['question']
            # for choice in self.choices:
            #     example += f'\n{choice}. {line[f"{choice}"]}'
            if include_answer:
                example += line["answer"]
            return example

    def __getitem__(self, idx):
        data = self.post_list[idx]
        subject_name = data['subject_name']
        question = self.format_example(data, include_answer=False)
        prefix = f"The following are multiple choice questions (with answers) about {subject_name}."

        history = []
        if "chatglm" in self.model_name_or_path.lower():
            sep = "\n\n" if "chatglm2" in self.model_name_or_path.lower() else "\n"
            offset = 1 if "chatglm2" in self.model_name_or_path.lower() else 0
            # Few-Shot example construction
            if hasattr(self, "dev_list"):
                history.append(prefix)
                k = self.args.max_few_shot
                dev_list = self.dev_list[subject_name]
                for i in range(min(k, len(dev_list))):
                    prompt, answer = self.format_example(dev_list[i], include_answer=True)
                    prompt = f"[Round {i + offset}]{sep}问：{prompt}{sep}答：{answer}"
                    history.append(prompt)
            # Concat few-shot/zero-shot examples with question.
            # If length of full prompt exceeds max_length, remove examples until the length is smaller than max_length
            question = f"[Round {len(history) + offset}]{sep}问：{question}{sep}答："
            while True:
                full_prompt = sep.join(history + [question])
                input_ids = self.tokenizer.encode(full_prompt)
                if len(input_ids) <= self.max_length:
                    break
                elif len(history) <= 1:
                    full_prompt = question
                    break
                else:
                    history.pop(-1)

            encoded_dict = self.tokenizer(full_prompt, max_length=self.max_length, return_tensors="pt",
                                          truncation="longest_first")
        elif "qwen" in self.model_name_or_path.lower():
            # Few-Shot example construction
            if hasattr(self, "dev_list"):
                k = self.args.max_few_shot
                dev_list = self.dev_list[subject_name]
                for i in range(min(k, len(dev_list))):
                    history.append(self.format_example(dev_list[i], include_answer=True))
            full_prompt, input_ids = qwen_make_context(self.tokenizer, question, history, system=prefix,
                                                       max_window_size=self.max_length)
            encoded_dict = {"input_ids": torch.tensor(input_ids, dtype=torch.int64)}
        else:
            # Few-Shot example construction
            if hasattr(self, "dev_list"):
                history.append(prefix)
                k = self.args.max_few_shot
                dev_list = self.dev_list[subject_name]
                for i in range(min(k, len(dev_list))):
                    history.append(self.format_example(dev_list[i], include_answer=True))
            # Concat few-shot/zero-shot examples with question.
            # If length of full prompt exceeds max_length, remove examples until the length is smaller than max_length
            while True:
                full_prompt = "\n\n".join(history + [question])
                input_ids = self.tokenizer.encode(full_prompt)
                if len(input_ids) <= self.max_length:
                    break
                elif len(history) <= 1:
                    full_prompt = question
                    break
                else:
                    history.pop(-1)

            encoded_dict = self.tokenizer(full_prompt, max_length=self.max_length, return_tensors="pt",
                                          truncation="longest_first")

        return {
            "input_ids": encoded_dict["input_ids"],
            "attention_mask": encoded_dict.get("attention_mask", None),
            "number_of_shots": max(len(history) - 1, 0),
            "subject_name_key": data['subject_name_key'],
            "answer": data.get('answer', None)
        }

    def load_dataset(self, filename, return_format="list"):
        datasets = list() if return_format == "list" else dict()
        dt = os.path.basename(filename)

        for subject_name_key, subject in self.subject_mapping.items():
            subject_name = subject[0]
            if isinstance(datasets, dict):
                datasets[subject_name] = list()
            dev_file_path = os.path.join(filename, f'{subject_name_key}_{dt}.csv')
            dev_df = pd.read_csv(dev_file_path, names=["question", "A", "B", "C", "D", "answer"])
            for i, val in dev_df.iterrows():
                d = val.to_dict()
                if isinstance(datasets, dict):
                    datasets[subject_name].append(d)
                else:
                    d['subject_name'] = subject_name
                    d['subject_name_key'] = subject_name_key
                    datasets.append(d)

        print_rank_0(f"Finished loading {dt} dataset")

        return datasets
