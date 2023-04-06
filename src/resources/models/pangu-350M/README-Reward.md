Link to github: [here](https://github.com/sunzeyeah/RLHF)

---

# Model Description

Pangu-α is proposed by a joint technical team headed by PCNL. It was first released in [this repository](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha)  It is the first large-scale Chinese pre-trained language model with 200 billion parameters trained on 2048 Ascend processors using an automatic hybrid parallel training strategy. The whole training process is done on the “Peng Cheng Cloud Brain II” computing platform with the domestic deep learning framework called MindSpore. The PengCheng·PanGu-α pre-training model can support rich applications, has strong few-shot learning capabilities, and has outstanding performance in text generation tasks such as knowledge question and answer, knowledge retrieval, knowledge reasoning, and reading comprehension.

This repository contains PyTorch implementation of PanGu model with 350 million parameters pretrained weights (FP32 precision). It uses supervised finetuned [pangu-350M-sft](https://huggingface.co/sunzeyeah/pangu-350M-sft) and performs **reward training** on [Chinese Chatgpt Corpus](https://huggingface.co/datasets/sunzeyeah/chinese_chatgpt_corpus).
