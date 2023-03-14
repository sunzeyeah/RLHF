## Features

---
提供2大功能：
- LLM模型评测：参考GPT类模型，基于ZeroShot和FewShot实现 
- ChatGPT模型训练pipeline：根据[Learning to Summarize from human feedback](https://arxiv.org/abs/2009.01325) ，实现3大流程: SFT、Reward Model和RLHF

## Setup

---
1. Install apex
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .  2>&1 | tee build.log
```

2. Install deepspeed
```bash
git clone https://github.com/microsoft/deepspeed
cd deepspeed
rm -rf build
TORCH_CUDA_ARCH_LIST="7.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check
```
根据下图，调整```TORCH_CUDA_ARCH_LIST="7.0"```为对应的NVIDIA GPU架构
![image info](./images/torch_cuda_list.png "torch_cuda_list")

3. install trlx
```bash
pip install -r requirements.txt
git clone https://github.com/CarperAI/trlx.git
cd trlx
pip install -e .
```

## Data & Model Download

---

1. 模型下载

| 模型      | size | 百度网盘地址  |  提取码      | 
| ----------- | ----------- | ----------- |  ----------- |
| [Pangu-350M](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha)   | 659MB | [Pangu-350M](https://pan.baidu.com/s/1IzgtW48S2PKyjxPPMe1rAQ) |  c5jj |
| [Pangu-2.6B](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha)   | 9.8GB | [Pangu-2.6B](https://pan.baidu.com/s/1Tzvja4q_LgQOwkWPQ4jShw)    | 2rad |
| [Pangu-13B](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha)   | 23.6GB | [Pangu-13B](https://pan.baidu.com/s/11fWAeYYKqI7pH0UiuJ5jEQ)    | u3dx |
| [GLM-335M-chinese](https://github.com/THUDM/GLM) | 679MB | [GLM-335M-chinese](https://pan.baidu.com/s/11Lef-E7Tsz5OGOueCpiqaA) | ii8e |
| [GLM-10B-chinese](https://github.com/THUDM/GLM)   |  |      |   |

2. 数据下载

| 数据集      | size | 百度网盘地址  |  提取码      | 
| ----------- | ----------- | ----------- |  ----------- |
| CLUE Benchmark   | 500MB | [CLUE Benchmark](https://pan.baidu.com/s/15F60nRbBd6d6UvyXdwbXQA) |  m6gt |
| SFT & Reward Data  | 4GB | [SFT & Reward Data](https://pan.baidu.com/s/1QRxtNZYTd2N_zOwqzfzvRw) |  ueiy |

SFT & Reward Data基于百科、知道问答、对联、古文、古诗词、微博数据构造，用于SFT和Reward阶段训练


## Usage

---
#### LLM模型评测

```bash
cd examples
bash eval_pretrain.sh
```

#### SFT

```bash
cd examples
bash eval_pretrain.sh
```
#### Reward Molde

```bash
cd examples
bash train_reward.sh
```

#### RLHF

```bash
cd examples
bash train_rlhf.sh
```
