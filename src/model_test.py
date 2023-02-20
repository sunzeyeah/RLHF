import os
import pathlib
from typing import List
import json
import torch

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import TextGenerationPipeline, AutoTokenizer, AutoModelForCausalLM

from src.utils import logger
from src.models.reward import GPTRewardModel


REWARD_CHECKPOINT_PATH = "reward_model/rm_checkpoint/checkpoint-3625/pytorch_model.bin"

if not os.path.exists(REWARD_CHECKPOINT_PATH):
    os.makedirs("resources/config/reward_model/rm_checkpoint", exist_ok=True)
    os.system(
        f"wget -O {REWARD_CHECKPOINT_PATH} \
        https://huggingface.co/CarperAI/openai_summarize_tldr_rm_checkpoint/resolve/main/pytorch_model.bin"
    )
SFT_MODEL_PATH = "sft/gptj-supervised-summarize-checkpoint/checkpoint-20000"



def load_dataset(path, split, max_samples):
    discard = 0
    datasets = []
    with open(f"{path}/web_text_zh_{split}.json",encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f),"load_dataset"):
            item = json.loads(line)
            sample = {"prompt": item['title'] + item['desc'] + "模型回答：",
                      "label":item["content"] }

            if len( sample['prompt'] + sample['label'])> 500:
                discard+=1
            else:
                datasets.append(sample)
            if i-discard+1 == max_samples:
                logger.info(f"File: {path}/web_text_zh_{split}.json Num of over-length: {discard}")
                return datasets
    logger.info(f"File: {path}/web_text_zh_{split}.json Num of over-length: {discard}")
    return datasets


if __name__ == "__main__":

    # model_path = "CPM_chk"
    model_path = "sft/CPM_dialogue/checkpoint-400"

    data_path = "resources/dialogue_dir"

    # # Load the pre-trained reward model
    # rw_tokenizer = AutoTokenizer.from_pretrained("chinese_gpt_chk")
    # ## for bert tokenizer
    # # rw_tokenizer.add_special_tokens({'eos_token': "<|endoftext|>"})
    # # rw_tokenizer.add_special_tokens({'bos_token': "<|startoftext|>"})
    # # rw_tokenizer.pad_token = rw_tokenizer.eos_token
    # ##
    #
    # rw_model = GPTRewardModel(SFT_MODEL_PATH,rw_tokenizer)
    # rw_model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH,map_location="cpu"))
    #
    # ###
    # assert rw_tokenizer.pad_token_id == rw_tokenizer.eos_token_id
    # rw_model.config.end_token_id = rw_tokenizer.eos_token_id
    # rw_model.config.pad_token_id = rw_model.config.eos_token_id
    # rw_model.config.bos_token_id = rw_tokenizer.bos_token_id
    # rw_model.config.eos_token_id = rw_tokenizer.eos_token_id
    # ###
    #
    #
    # rw_model.half()
    # rw_model.eval()
    #
    # rw_device = torch.device("cuda:{}".format(7) if torch.cuda.is_available() else "cpu")  # set reward model device
    #
    #
    # rw_model.to(rw_device)


    tokenizer = AutoTokenizer.from_pretrained("CPM_chk")


    def get_scores(samples: List[str]):# ...<text>... TL;DR: <text>
        scores_list = []
        batch_size = 2
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i : i + batch_size]
            sub_samples = ["<|startoftext|>" + chosen.replace("模 型 回 答 ：","") + "<|endoftext|>" for chosen in sub_samples]
            encodings_dict = rw_tokenizer(
                sub_samples,
                truncation=True,
                max_length=450,
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
            if i==0:
                logger.info("line:108", prompts[0])

            tmp = tokenizer.decode(
                tokenizer(prompts[i], truncation=True, max_length=max_length)["input_ids"],
                skip_special_tokens=True,
            ).strip()
            # tmp = prompts[i] + "\nTL;DR:"
            formatted_prompts.append(tmp)

        return formatted_prompts

    def reward_fn(samples: List[str], **kwargs):# ...<text>... TL;DR: <text>
        logger.info("Line 122: samples",samples[0])
        original_samples = [text.split("模 型 回 答 ：")[0] + "模 型 回 答 ：" for text in samples]
        # logger.info("original_samples",original_samples)

        original_samples = [text + post_summary_dict[text.strip()] for text in original_samples]
        original_scores = get_scores(original_samples)
        scores = get_scores(samples)
        norms_scores = scores - original_scores
        return norms_scores



    # Store data into prompt and label pairs
    train_set = [(sample["prompt"], sample["label"]) for sample in  load_dataset(data_path,"valid",200)]

    # Split contents into summaries and labels
    train_posts, train_summaries = zip(*train_set)


    # Get the OpenAI summaries
    post_summary_dict = {}
    train_prompts = get_prompt_dataset(train_posts, 450)
    for i in range(len(train_prompts)):
        post_summary_dict[train_prompts[i]] = train_summaries[i]


    # logger.info("###### post_summary_dict ########")
    # for ik,query in enumerate(post_summary_dict):
    #     if ik==10:
    #         break
    #     logger.info(query)
    #     logger.info(post_summary_dict[query])
    #     logger.info("\n")
    # logger.info("###### post_summary_dict ########")

    tokenizer.padding_side = "left"
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    text_generator = TextGenerationPipeline(model, tokenizer, device=device)
    logger.info(f"load from {model_path}")

    for j in range(5):
        prompt = train_posts[j]
        prompt_dict = tokenizer(
            prompt,
            truncation=True,
            max_length=450,
            padding="max_length",
            return_tensors="pt",
        )

        gen = model.generate(input_ids = prompt_dict['input_ids'].to(device),
                           attention_mask = prompt_dict['attention_mask'].to(device),
                           max_length=512,
                            do_sample=True,
                            top_p=0.9
                           )
        out = tokenizer.decode(gen[0, :], skip_special_tokens=True)

        out2=text_generator(prompt, max_length=512, do_sample=True, top_p=0.9)

        logger.info("model.generate",out)
        logger.info("text_generator",out2)

        logger.info("label",train_summaries[j])

        logger.info("\n")


    #
    #
    #
    # logger.info(prompt)
    # logger.info(prompt_dict)
    # logger.info(out)

