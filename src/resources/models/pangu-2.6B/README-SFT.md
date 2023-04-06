Link to github: [here](https://github.com/sunzeyeah/RLHF)

---

# Model Description

Pangu-α is proposed by a joint technical team headed by PCNL. It was first released in [this repository](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha)  It is the first large-scale Chinese pre-trained language model with 200 billion parameters trained on 2048 Ascend processors using an automatic hybrid parallel training strategy. The whole training process is done on the “Peng Cheng Cloud Brain II” computing platform with the domestic deep learning framework called MindSpore. The PengCheng·PanGu-α pre-training model can support rich applications, has strong few-shot learning capabilities, and has outstanding performance in text generation tasks such as knowledge question and answer, knowledge retrieval, knowledge reasoning, and reading comprehension.

This repository contains PyTorch implementation of PanGu model with 2.6 billion parameters pretrained weights (FP32 precision). It uses pretrained [pangu-2.6B](https://huggingface.co/imone/pangu_2_6B) model and performs **supervised finetuning (SFT)** on [Chinese Chatgpt Corpus](https://huggingface.co/datasets/sunzeyeah/chinese_chatgpt_corpus).

---

# Usage (Text Generation)

Currently PanGu model is not supported by transformers,
so `trust_remote_code=True` is required to load model implementation in this repo.

```python
from transformers import TextGenerationPipeline, AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("sunzeyeah/pangu-2.6B-sft", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("sunzeyeah/pangu-2.6B-sft", trust_remote_code=True)
prompt = "我不能确定对方是不是喜欢我,我却想分分秒秒跟他在一起,有谁能告诉我如何能想他少一点<sep>回答："
inputs = tokenizer(prompt, add_special_tokens=False, return_token_type_ids=False, return_tensors="pt")
outputs = model.generate(**inputs,
                         max_new_tokens=100,
                         pad_token_id=tokenizer.pad_token_id,
                         do_sample=False,
                         num_return_sequences=1,
                         top_p=0.8,
                         temperature=0.8)
results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
results = [result.split("答:", maxsplit=1)[1] for result in results]
print(results)
```

Expected output:
```python
["你爱他就多关心他,少在他面前表现出你的脆弱。这样他才更会爱你。"]
```
