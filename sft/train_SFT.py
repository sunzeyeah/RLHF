import random

import evaluate
import numpy as np
import torch
from Daiglogue_dataset import TLDRDataset
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
    output_dir = "D:\\Data\\output\\pangu-350M"
    train_batch_size = 2
    gradient_accumulation_steps = 16
    learning_rate = 1e-5
    eval_batch_size = 1
    eval_steps = 500
    max_input_length = 512
    save_steps = 1000
    num_train_epochs = 1
    random.seed(42)

    model_name_or_path = "D:\\Data\\models\\pangu-350M"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_cache=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_cache=False, trust_remote_code=True)

    print(f"Finished loading model and tokenizer from {model_name_or_path}")

    ## for bert tokenizer
    tokenizer.add_special_tokens({'eos_token': "<eot>", 'pad_token': "<eot>"})
    # tokenizer.add_special_tokens({'bos_token': "<s>"})

    model.resize_token_embeddings(len(tokenizer.sp))
    assert tokenizer.pad_token_id == tokenizer.eos_token_id

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id


    # Set up the datasets
    data_path = "D:\\Data\\raw"
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
        gradient_checkpointing=False,
        half_precision_backend="auto",
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
        report_to=["tensorboard"],
        # deepspeed="./ds_config_pangu.json",
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
