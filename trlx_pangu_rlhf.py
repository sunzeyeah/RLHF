import os
import pathlib
from typing import List
import json
import torch
<<<<<<< HEAD

from reward_model.reward_model import GPTRewardModel
from tqdm import tqdm
from transformers import AutoTokenizer

import trlx
from trlx.data.configs import TRLConfig

import datetime
import logging

def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

logging.Formatter.converter = beijing

logging.basicConfig(
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


REWARD_CHECKPOINT_PATH = "./pytorch_model.bin"


SFT_MODEL_PATH = "./Pangu_chk"

T_PATH = "imone/pangu_2_6B"

=======
from reward_model.reward_model import GPTRewardModel
from tqdm import tqdm
from transformers import AutoTokenizer,AutoConfig
import trlx
from trlx.data.configs import TRLConfig
import datetime
import logging
def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()
logging.Formatter.converter = beijing
logging.basicConfig(
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S")

##################################################################################################################

SFT_MODEL_PATH = "<SFT Pangu path>"
SFT_Data_PATH = "<SFT dataset path>"
RM_Token_PATH = "<reward model tokenizer path>"
RM_Model_PATH = "<reward model checkpoint path>"
GPT_Token_PATH = "<Pangu tokenizer path>"

##################################################################################################################
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1

def load_dataset(path, split, max_samples):
    discard = 0
    datasets = []
    with open(f"{path}/web_text_zh_{split}.json",encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f),"load_dataset"):
            item = json.loads(line)
            sample = {"prompt": item['title'] + item['desc'] + "模型回答:",
                      "label":item["content"] }

            if len( sample['prompt'] + sample['label'])> 500:
                discard+=1
            else:
                datasets.append(sample)
            if i-discard+1 == max_samples:
                logging.info(f"File: {path}/web_text_zh_{split}.json Num of over-length: {discard}")
                return datasets
    logging.info(f"File: {path}/web_text_zh_{split}.json Num of over-length: {discard}")
    return datasets


if __name__ == "__main__":

<<<<<<< HEAD
    model_path = "chinese_gpt_chk"
    token_path = "tokenizer_chk"
    data_path = "./dialogue_dir"

    # Load the pre-trained reward model
    rw_tokenizer = AutoTokenizer.from_pretrained(token_path)
    rw_model = GPTRewardModel("./checkpoint-20000",rw_tokenizer)
    rw_model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH))

    ###
=======
    # Prepare RM
    rw_tokenizer = AutoTokenizer.from_pretrained(RM_Token_PATH)

    rw_config = AutoConfig.from_pretrained(RM_Model_PATH)
    rw_model = GPTRewardModel(rw_config,rw_tokenizer)
    rw_model.load_state_dict(torch.load(f"{RM_Model_PATH}/pytorch_model.bin"))
    logging.info(f"Now, the RM is reloaded from {RM_Model_PATH}/pytorch_model.bin")


    # Prepare RM config
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1
    assert rw_tokenizer.pad_token_id == rw_tokenizer.eos_token_id
    rw_model.config.end_token_id = rw_tokenizer.eos_token_id
    rw_model.config.pad_token_id = rw_model.config.eos_token_id
    rw_model.config.bos_token_id = rw_tokenizer.bos_token_id
    rw_model.config.eos_token_id = rw_tokenizer.eos_token_id
<<<<<<< HEAD
    ###

=======
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1

    rw_model.half()
    rw_model.eval()
    rw_device = torch.device("cuda:{}".format(7))  # set reward model device
<<<<<<< HEAD


    rw_model.to(rw_device)
=======
    rw_model.to(rw_device)
    ###
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1

    def get_scores(samples: List[str]):# ...<text>... TL;DR: <text>
        scores_list = []
        batch_size = 2
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i : i + batch_size]
            sub_samples = [rw_tokenizer.bos_token + chosen + rw_tokenizer.eos_token for chosen in sub_samples]
            encodings_dict = rw_tokenizer(
                sub_samples,
                truncation=True,
                max_length=config.train.seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(rw_device)
            attn_masks = encodings_dict["attention_mask"].to(rw_device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            scores_list.append(sub_scores["chosen_end_scores"])
        scores = torch.cat(scores_list, dim=0)
        return scores

    def get_prompt_dataset(prompts, max_length):# ...<text>... TL;DR:
        """
        Get the prompt after T5 decoding to make sure dictionary
        of prompts and summaries is consistent decode prompt from trlX pipeline
        """
        formatted_prompts = []
        for i in tqdm(range(len(prompts)),"get_prompt_dataset"):
            tmp = tokenizer.decode(
                tokenizer(prompts[i], truncation=True, max_length=max_length)["input_ids"],
                skip_special_tokens=True,
            ).strip()
            # tmp = prompts[i] + "\nTL;DR:"
            formatted_prompts.append(tmp)
<<<<<<< HEAD

=======
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1
        return formatted_prompts

    def reward_fn(samples: List[str], **kwargs):# ...<text>... TL;DR: <text>

<<<<<<< HEAD
        for i in range(5):
            if i < len(samples):
                logging.info(f"Line 122: demo samples-{i+1} {samples[0]}")
=======
        rank =  torch.distributed.get_rank()
        if rank==0:
            logging.info(f"[rank-{rank}]: {samples[0]}")
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1

        original_samples = [text.split("模型回答:")[0] + "模型回答:" for text in samples]
        original_samples = [text + post_summary_dict[text.strip()] for text in original_samples]
        original_scores = get_scores(original_samples)
        scores = get_scores(samples)
        norms_scores = scores - original_scores
        return norms_scores

    config_path = pathlib.Path(__file__).parent.joinpath("configs/ppo_config_summ_gptj.yml")
    config = TRLConfig.load_yaml(config_path)
<<<<<<< HEAD

    config.model.model_path = SFT_MODEL_PATH
    config.tokenizer.tokenizer_path = T_PATH

    tokenizer=AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(config)

    max_length_input = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    # logging.info(f"max_length_input = {max_length_input}")


    # Store data into prompt and label pairs
    train_set = [(sample["prompt"], sample["label"]) for sample in  load_dataset(data_path,"train",200000)]
    val_set = [(sample["prompt"], sample["label"])  for sample in  load_dataset(data_path,"valid",2000)]
=======
    config.model.model_path = SFT_MODEL_PATH
    config.tokenizer.tokenizer_path = GPT_Token_PATH
    tokenizer=AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(config)
    max_length_input = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    # Store data into prompt and label pairs
    train_set = [(sample["prompt"], sample["label"]) for sample in  load_dataset(SFT_Data_PATH,"train",200000)]
    val_set = [(sample["prompt"], sample["label"])  for sample in  load_dataset(SFT_Data_PATH,"valid",2000)]
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1

    # Split contents into summaries and labels
    train_posts, train_summaries = zip(*train_set)
    val_posts, val_summaries = zip(*val_set)

    # Get the OpenAI summaries
    post_summary_dict = {}
    train_prompts = get_prompt_dataset(train_posts, max_length_input)
    for i in range(len(train_prompts)):
        post_summary_dict[train_prompts[i]] = train_summaries[i]
    val_prompts = get_prompt_dataset(val_posts, max_length_input)
    for i in range(len(val_prompts)):
        post_summary_dict[val_prompts[i]] = val_summaries[i]

<<<<<<< HEAD
    # logging.info("\n###### post_summary_dict ########")
    # for ik,query in enumerate(post_summary_dict):
    #     if ik==3:
    #         break
    #     logging.info(query)
    #     logging.info(post_summary_dict[query])
    # logging.info("###### post_summary_dict ########\n")

=======
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1
    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts[0:1000],  # sampling 1000 validation prompts for evaluation speed in training
        config=config,
    )
