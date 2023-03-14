## Features

提供2大功能：
- LLM模型评测：参考GPT类模型，基于ZeroShot和FewShot实现 
- ChatGPT模型训练pipeline：根据[Learning to Summarize from human feedback](https://arxiv.org/abs/2009.01325) ，实现3大流程: SFT、Reward Model和RLHF

## Setup

### 1. Install apex
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .  2>&1 | tee build.log
```

### 2. Install deepspeed
```bash
git clone https://github.com/microsoft/deepspeed
cd deepspeed
rm -rf build
TORCH_CUDA_ARCH_LIST="7.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check
```
根据下图，调整```TORCH_CUDA_ARCH_LIST="7.0"```为对应的NVIDIA GPU架构
![image info](./images/torch_cuda_list.png "torch_cuda_list")

### 3. install trlx
```bash
pip install -r requirements.txt
git clone https://github.com/CarperAI/trlx.git
cd trlx
pip install -e .
```

## Data & Model Download

### 1. 模型下载

| 模型      | size | 百度网盘地址  |  提取码      | 
| ----------- | ----------- | ----------- |  ----------- |
| [Pangu-350M](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha)   | 659MB | [Pangu-350M](https://pan.baidu.com/s/1IzgtW48S2PKyjxPPMe1rAQ) |  c5jj |
| [Pangu-2.6B](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha)   | 9.8GB | [Pangu-2.6B](https://pan.baidu.com/s/1Tzvja4q_LgQOwkWPQ4jShw)    | 2rad |
| [Pangu-13B](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha)   | 23.6GB | [Pangu-13B](https://pan.baidu.com/s/11fWAeYYKqI7pH0UiuJ5jEQ)    | u3dx |
| [GLM-335M-chinese](https://github.com/THUDM/GLM) | 679MB | [GLM-335M-chinese](https://pan.baidu.com/s/11Lef-E7Tsz5OGOueCpiqaA) | ii8e |
| [GLM-10B-chinese](https://github.com/THUDM/GLM)   |  |      |   |

### 2. 数据下载

| 数据集      | size | 百度网盘地址  |  提取码      | 
| ----------- | ----------- | ----------- |  ----------- |
| [CLUE Benchmark](https://github.com/CLUEbenchmark/CLUE)   | 500MB | [CLUE Benchmark](https://pan.baidu.com/s/15F60nRbBd6d6UvyXdwbXQA) |  m6gt |
| SFT & Reward Data  | 4GB | [SFT & Reward Data](https://pan.baidu.com/s/1QRxtNZYTd2N_zOwqzfzvRw) |  ueiy |
| [百科](https://github.com/brightmart/nlp_chinese_corpus)  | 652MB | [baike_qa_2019](https://pan.baidu.com/s/1N6I-fvx6FLHwuxJuDLLA8g) | 7jad |
| [知道问答](https://github.com/SophonPlus/ChineseNlpCorpus) | 847MB | [zhidao](https://pan.baidu.com/s/1sjR3vABiMbdV1HkUt6kCKQ) | neds |
| [对联](https://github.com/wb14123/couplet-dataset/releases/download/1.0/couplet.tar.gz)  | 221MB | [couplets](https://pan.baidu.com/s/1oo6oaephPXpyU-sDd_37qg) | 54ey |
| [古文](https://github.com/NiuTrans/Classical-Modern)  | 125MB | [Classical & Modern](https://pan.baidu.com/s/1ZyGkYOgzT9ZEVnpSpB4kQg) | a4cr |
| [古诗词](https://github.com/chinese-poetry/chinese-poetry)  | 87MB | [chinese poetry](https://pan.baidu.com/s/13uvkA96PdKpKB7ZQ2GkXZQ) | 5zzj |
| 微博新闻评论  | 522MB | [weibo summary comments](https://pan.baidu.com/s/1h45O0q6gQl3LbH-NjzuRlw) | w0g1 |

**PS**: SFT & Reward Data基于百科、知道问答、对联、古文、古诗词、微博新闻评论数据构造，可直接用于SFT和Reward阶段训练，详见[data_prepare.py](./src/data_prepare.py)


## Usage

### 1. LLM模型评测
对开源中文LLM进行ZeroShot、OneShot或FewShot的评测，评测任务和数据集使用[CLUEBenchmark](https://github.com/CLUEbenchmark/CLUE) 。评测方法和prompt模板参考[Pangu-alpha论文](https://arxiv.org/abs/2104.12369) ，详见[eval_pretrain.py](./src/eval_pretrain.py) 和 [data.py](./src/utils/data.py)

目前支持5个开源模型: 
- Pangu-350M
- Pangu-2.6B
- Pangu-13B
- GLM-335M-chinese
- GLM-10B-chinese

```bash
cd examples
bash eval_pretrain.sh
```

### 2. SFT
使用开源LLM + SFT&Reward数据进行SFT训练
```bash
cd examples
bash train_sft.sh
```
### 3. Reward Model
使用SFT模型 + SFT&Reward数据进行Reward模型训练。训练时，将SFT模型的前70%层固定，不进行梯度更新
```bash
cd examples
bash train_reward.sh
```

### 4. RLHF
基于PPO算法进一步训练SFT模型
```bash
cd examples
bash train_rlhf.sh
```


## Results

### 1. LLM模型评测
以下为验证集(dev.json)结果：

<table>
    <tr>  <td rowspan="2">Dataset</td>  <td rowspan="2">Method</td>  <td rowspan="2">Metrics</td>  <td rowspan="2">Task Type</td>  <td colspan="5" style="text-align:center">Zero-shot</td>  <td colspan="5" style="text-align:center">Few-shot</td> </tr>
    <tr>  <td>Pangu-350M</td>  <td>Pangu-2.6B</td>  <td>Pangu-13B</td>  <td>GLM-335M-chinese</td>  <td>GLM-10B-chinese</td>  <td>Pangu-350M</td>  <td>Pangu-2.6B</td>  <td>Pangu-13B</td>  <td>GLM-335M-chinese</td>  <td>GLM-10B-chinese</td> </tr>
    <tr>  <td>OCNLI</td>  <td>PPL</td>  <td>acc</td>  <td>NLI</td>  <td>0.3369</td>  <td>0.3061</td>  <td>0.3074</td>  <td></td>  <td></td>  <td>0.3352</td>  <td>0.3216</td>  <td>0.3298</td>  <td></td>  <td></td> </tr>
    <tr>  <td>CMNLI</td>  <td>PPL</td>  <td>acc</td>  <td>NLI</td>  <td>0.3302</td>  <td>0.3310</td>  <td>0.3279</td>  <td></td>  <td></td>  <td>0.3328</td>  <td>0.3300</td>  <td></td>  <td></td>  <td></td> </tr>
    <tr>  <td>CHID</td>  <td>PPL</td>  <td>acc</td>  <td>Cloze(multi-choices)</td>  <td>0.0916</td>  <td>0.0670</td>  <td>0.0734</td>  <td></td>  <td>0.1016</td>  <td></td>  <td></td>  <td></td>  <td></td>  <td></td> </tr>
    <tr>  <td>CMRC2018</td>  <td>generation</td>  <td>f1</td>  <td>MRC</td>  <td>0.0979</td>  <td>0.1007</td>  <td>0.093</td>  <td></td>  <td>0.1392</td>  <td></td>  <td></td>  <td></td>  <td></td>  <td></td> </tr>
    <tr>  <td>CLUEWSC2020</td>  <td>PPL</td>  <td>acc</td>  <td>WSC</td>  <td>0.5328</td>  <td>0.5592</td>  <td>0.4934</td>  <td></td>  <td>0.5131</td>  <td>0.4473</td>  <td>0.4671</td>  <td>0.5526</td>  <td></td>  <td></td> </tr>
    <tr>  <td>C3</td>  <td>PPL</td>  <td>acc</td>  <td>Common sense reasoning</td>  <td>0.2426</td>  <td>0.2418</td>  <td>0.2360</td>  <td></td>  <td>0.2573</td>  <td></td>  <td></td>  <td></td>  <td></td>  <td></td> </tr>
    <tr>  <td>AFQMC</td>  <td>PPL</td>  <td>acc</td>  <td>Text classification</td>  <td>0.4582</td>  <td>0.4914</td>  <td>0.6306</td>  <td></td>  <td>0.4960</td>  <td>0.4993</td>  <td>0.5018</td>  <td>0.4872</td>  <td></td>  <td></td> </tr>
    <tr>  <td>CSL</td>  <td>PPL</td>  <td>acc</td>  <td>Text classification</td>  <td>0.4913</td>  <td>0.4666</td>  <td>0.4943</td>  <td></td>  <td>0.5126</td>  <td>0.5036</td>  <td>0.4973</td>  <td>0.514</td>  <td></td>  <td></td> </tr>
    <tr>  <td>IFLYTEK</td>  <td>PPL</td>  <td>acc</td>  <td>Text classification</td>  <td>0.3058</td>  <td>0.265</td>  <td>0.1292</td>  <td></td>  <td>0.2620</td>  <td>0.2535</td>  <td>0.2524</td>  <td>0.2539</td>  <td></td>  <td></td> </tr>
    <tr>  <td>TNEWS</td>  <td>PPL</td>  <td>acc</td>  <td>Text classification</td>  <td>0.2022</td>  <td>0.2449</td>  <td>0.1582</td>  <td></td>  <td>0.2489</td>  <td></td>  <td></td>  <td></td>  <td></td>  <td></td> </tr>
</table>


### 2. SFT

| 模型 | 硬件 | batch size | sequence length | gpu memory used | speed |
| --- | --- | :---: | :---: | :---: | --- |
| Pangu-2.6B | A100 80G | 8 | 512 | 79.4G | 9.61 s/iter


### 3. Reward Model
以Pang-2.6B模型为例，在单张A100(80G)的训练结果如下:

| |	SFT	| Reward |
| --- | --- | --- |
| # trainable params | 2.6B |	815M |
| # samples	| 5.4M	| 12M |
| Hours per epoch	| 116h	| 423h |
| Batch size	| 8	| 8 |
| GPU Memory used	| 79.4G	| 80.7G |

