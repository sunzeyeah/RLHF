import random

import evaluate
import numpy as np
import torch
<<<<<<< HEAD
from summarize_dataset import TLDRDataset
=======
from Daiglogue_dataset import TLDRDataset
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


if __name__ == "__main__":
<<<<<<< HEAD
    output_dir = "CPM_dialogue"
=======
    output_dir = "Pangu_dialogue"
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1
    train_batch_size = 48
    gradient_accumulation_steps = 2
    learning_rate = 1e-5
    eval_batch_size = 1
    eval_steps = 500
    max_input_length = 512
    save_steps = 1000
    num_train_epochs = 5
    random.seed(42)

<<<<<<< HEAD
    downloadmodel_path = "/userhome/Research_HUB/RLHF/trlx/examples/dialogue_rlhf/CPM_chk"
    tokenizer = AutoTokenizer.from_pretrained(downloadmodel_path)
    model_path =  "/userhome/Research_HUB/RLHF/trlx/examples/dialogue_rlhf/sft/CPM_dialogue/checkpoint-3000"
=======
    downloadmodel_path = "Pangu_chk"
    tokenizer = AutoTokenizer.from_pretrained(downloadmodel_path)
    model_path =  "/sft/Pangu_dialogue/checkpoint"
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1
    model = AutoModelForCausalLM.from_pretrained(model_path, use_cache=False)

    print(f"load model from {model_path}")

    ## for bert tokenizer
    # tokenizer.add_special_tokens({'eos_token': "<|endoftext|>"})
    # tokenizer.add_special_tokens({'bos_token': "<|startoftext|>"})
    ##

    tokenizer.pad_token = tokenizer.eos_token

    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer.pad_token_id == tokenizer.eos_token_id

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id


    # Set up the datasets
    # data_path = "CarperAI/openai_summarize_tldr"
<<<<<<< HEAD
    data_path = "/userhome/Research_HUB/RLHF/trlx/examples/dialogue_rlhf/dialogue_dir"
=======
    data_path = "dialogue_dir"
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1
    train_dataset = TLDRDataset(
        data_path,
        tokenizer,
        "train",
        max_length=max_input_length,
    )
    dev_dataset = TLDRDataset(
        data_path,
        tokenizer,
        "valid",
        max_length=max_input_length,
    )

    # Set up the metric
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        labels_ids = eval_preds.label_ids
        pred_ids = eval_preds.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        result = rouge.compute(predictions=pred_str, references=label_str)
        return result

    # Create a preprocessing function to extract out the proper logits from the model output
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    # Prepare the trainer and start training
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_accumulation_steps=1,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_checkpointing=True,
        half_precision_backend=True,
        fp16=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        warmup_steps=100,
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        logging_steps=10,
<<<<<<< HEAD
        deepspeed="./ds_config_gptj.json",
=======
        deepspeed="./ds_config_pangu.json",
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)
