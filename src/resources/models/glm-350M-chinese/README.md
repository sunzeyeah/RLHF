---
language:
- zh
tags:
- glm
- chatgpt
---

Link to github: [here](https://github.com/sunzeyeah/RLHF)

---

本仓库由[THUDM/glm-large-chinese](https://huggingface.co/THUDM/glm-large-chinese) fork而来，原仓库实现了PyTorch版本的GLM模型，该模型有3.5亿参数量，模型权重文件以FP32格式存储。

本仓库在原始代码的基础上进行了部分调整，以支持ChatGPT训练pipeline，具体实现可参考：[sunzeyeah/RLHF](https://github.com/sunzeyeah/RLHF).

This repository is forked from [THUDM/glm-large-chinese](https://huggingface.co/THUDM/glm-large-chinese) that contains PyTorch implementation of GLM model with 350 million parameters pretrained weights (FP32 precision).

It is slightly different from the original GLM implementation to support the ChatGPT training pipeline in this github repo: [sunzeyeah/RLHF](https://github.com/sunzeyeah/RLHF).

---

# Model description
GLM is a General Language Model pretrained with an autoregressive blank-filling objective and can be finetuned on various natural language understanding and generation tasks.

Please refer to our paper for a detailed description of GLM:

[GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360) (ACL 2022)

Zhengxiao Du*, Yujie Qian*, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, Jie Tang (*: equal contribution)

Find more examples in our [Github repo](https://github.com/THUDM/GLM).

`glm-10b-chinese` is pretrained on the [WuDaoCorpora](https://www.sciencedirect.com/science/article/pii/S2666651021000152) dataset. It has 48 transformer layers, with hidden size 4096 and 64 attention heads in each layer. The model is pretrained with autoregressive blank filling objectives designed for natural language understanding, seq2seq, and language modeling.

---

# Usage (Text Generation)
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("sunzeyeah/glm-350M-chinese", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("sunzeyeah/glm-350M-chinese", trust_remote_code=True)
model = model.half().cuda()

max_length = 512
prompt = "我不能确定对方是不是喜欢我,我却想分分秒秒跟他在一起,有谁能告诉我如何能想他少一点"
prefix = "回答："
encoded_prompt = tokenizer(prompt, prefix + tokenizer.mask_token)
prompt_length = len(encoded_prompt['input_ids'])
encoded_dict = tokenizer(prompt, prefix + tokenizer.mask_token,
                         max_length=min(prompt_length, max_length),
                         truncation="only_first",
                         return_tensors="pt",
                         return_token_type_ids=False)
max_gen_length = max_length - encoded_dict['input_ids'].shape[1]
inputs = tokenizer.build_inputs_for_generation(encoded_dict, max_gen_length=max_gen_length, padding=True)
inputs = inputs.cuda()
outputs = model.generate(**inputs,
                         max_new_tokens=max_gen_length,
                         eos_token_id=tokenizer.eop_token_id,
                         pad_token_id=tokenizer.pad_token_id,
                         do_sample=False,
                         num_return_sequences=1,
                         top_p=0.8,
                         temperature=1.0)
results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(results)
```

