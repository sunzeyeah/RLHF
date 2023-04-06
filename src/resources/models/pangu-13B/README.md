---
language:
- zh
tags:
- pangu
- chatgpt
---

Link to github: [here](https://github.com/sunzeyeah/RLHF)

---


# Model Description

Pangu-α is proposed by a joint technical team headed by PCNL. It was first released in [this repository](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha)  It is the first large-scale Chinese pre-trained language model with 200 billion parameters trained on 2048 Ascend processors using an automatic hybrid parallel training strategy. The whole training process is done on the “Peng Cheng Cloud Brain II” computing platform with the domestic deep learning framework called MindSpore. The PengCheng·PanGu-α pre-training model can support rich applications, has strong few-shot learning capabilities, and has outstanding performance in text generation tasks such as knowledge question and answer, knowledge retrieval, knowledge reasoning, and reading comprehension.

This repository contains PyTorch implementation of PanGu model with 13 billion parameters pretrained weights (FP32 precision). 

It is slightly different from the [original pangu implementation](https://huggingface.co/imone/pangu_2_6B) to support the ChatGPT training pipeline in this github repo: [sunzeyeah/RLHF](https://github.com/sunzeyeah/RLHF).

---
