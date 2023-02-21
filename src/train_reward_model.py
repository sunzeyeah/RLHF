import os
import torch

# from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

from src.models.reward import GPTRewardModel


def load_dataset(path, split):
    d = torch.load(f"{path}/{split}_data.pt")
    for k in range(5):
        print(f"reward data demo-{k}:\n {d[-k-1]}")
    return d


def create_comparison_dataset(path="CarperAI/openai_summarize_comparisons", split="train"):
    dataset = load_dataset(path, split=split)
    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        prompt = sample["prompt"]
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary) < 5 or len(rejected_summary) < 5:
            continue
        pair["chosen"] = prompt + "\n" + chosen_summary
        pair["rejected"] = prompt + "\n" + rejected_summary
        pairs.append(pair)
    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        self.num_skip = 0
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                "<|startoftext|>" + chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                "<|startoftext|>" + rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )

            if torch.all(torch.eq(chosen_encodings_dict["input_ids"],rejected_encodings_dict["input_ids"])).item():
                self.num_skip+=1
                continue

            self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
            self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
            self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
            self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

        print(f"PairwiseDataset skip instances: {self.num_skip}")


    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch


def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result


if __name__ == "__main__":

    model_path = "../chinese_gpt_chk"
    # model_path = "../CPM_chk"
    # model_path = "imone/pangu_2_6B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer.pad_token = tokenizer.eos_token

    ## for bert tokenizer
    tokenizer.add_special_tokens({'eos_token': "<|endoftext|>"})
    tokenizer.add_special_tokens({'bos_token': "<|startoftext|>"})
    tokenizer.pad_token = tokenizer.eos_token
    ##


    if not os.path.exists("resources/config/reward_model/rm_checkpoint"):
        os.mkdir("resources/config/reward_model/rm_checkpoint")

    training_args = TrainingArguments(
        output_dir="resources/config/reward_model/rm_checkpoint/",
        num_train_epochs=5,
        logging_steps=10,
        gradient_accumulation_steps=2,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_accumulation_steps=1,
        eval_steps=125,
        save_steps=125,
        warmup_steps=100,
        logging_dir="./logs",
        fp16=True,
        bf16=False,
        learning_rate=1e-5,
        deepspeed="ds_config_gpt_j.json",
        save_total_limit=1,
    )

    # Initialize the reward model from the (supervised) fine-tuned GPT-J
    model = GPTRewardModel(model_path, tokenizer)



    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    # Create the comparisons datasets
    # data_path = "CarperAI/openai_summarize_comparisons"
    data_path = "resources/reward_data_dir/processed"

    train_pairs = create_comparison_dataset(data_path, "train")
    val_pairs = create_comparison_dataset(data_path, "test")

    # Make pairwise datasets for training
    max_length = 1024
    train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
    val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward()

    Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    ).train()