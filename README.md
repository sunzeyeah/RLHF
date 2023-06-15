## Features

提供3大功能：
- LLM模型预训练：支持常见模型的预训练，包括：decoder结构（LLaMA、GPT）、encoder结构（GLM）
- LLM模型评测：参考GPT类模型，基于ZeroShot和FewShot实现
- ChatGPT模型训练pipeline：根据[Learning to Summarize from human feedback](https://arxiv.org/abs/2009.01325) ，实现3大流程: SFT、Reward Model和RLHF

## Setup

### 1. Install deepspeed
```bash
git clone https://github.com/microsoft/DeepSpeed.git
cd deepspeed
rm -rf build
TORCH_CUDA_ARCH_LIST="7.0" DS_BUILD_OPS=1 pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1 | tee build.log
```
如果想创建binary wheel，方便在其他机器上安装，可使用如下命令，会在```dist```目录生成类似可安装文件```deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl```
```bash
git clone https://github.com/microsoft/DeepSpeed.git
cd deepspeed
rm -rf build
TORCH_CUDA_ARCH_LIST="7.0" DS_BUILD_OPS=1 python setup.py build_ext -j8 bdist_wheel 2>&1 | tee build.log
```
**PS**：需要根据下图，调整```TORCH_CUDA_ARCH_LIST="7.0"```为自己对应的NVIDIA GPU架构
![image info](./images/torch_cuda_list.png "torch_cuda_list")

或运行```torch.cuda.get_device_capability()```获取自己GPU的架构

### 2. Install jieba
在使用Pangu类模型的时候，其special_token格式为```<sep>```、```<pad>```等，而[tokenization_gptpangu.py](src/resources/models/pangu-350M/tokenization_gptpangu.py)中```tokenize()```函数会使用```jieba```进行分词。但直接```pip install jieba```，默认会将```<```和```>```直接切分开，使用```jieba.add_word("<sep>")```也没有作用，因为```jieba```直接hardcode了会自动切分的token，其中就包括了```<```和```>```。 

因此需要执行：
```bash
git clone https://github.com/fxsjy/jieba.git
cd jieba
```
将代码clone到本地，修改```jieba/__init__.py```中```re_han_default```的取值，具体改动如下：

- 改动前：
```python
re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)", re.U)
```

- 改动后：
```python
re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-<>]+)", re.U)
```

修改完成后使用```pip install .```进行本地编译安装，替换原有```jieba```。安装完成后，在代码中加入```jieba.add_word("<sep>")```（该代码已加入[tokenization_gptpangu.py](src/resources/models/pangu-350M/tokenization_gptpangu.py)），即可解决将```<sep>```一类的special token切分为多个id的情况


### 3. Install apex (Optional)
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .  2>&1 | tee build.log
```
如果想创建binary wheel，方便在其他机器上安装，可使用如下命令，会在```dist```目录生成类似可安装文件```apex-0.0.1+7150e20-cp38-cp38-linux_x86_64.whl```
```bash
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py --cpp_ext --cuda_ext bdist_wheel 2>&1 | tee build.log
```


## Data & Model Download

### 1. 预训练模型下载

| 模型      | size | huggingface地址 | 百度网盘地址  |  提取码      | 
| ----------- | ----------- | ----------- |  ----------- |  ----------- |
| [Pangu-350M](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha)   | 659MB | [sunzeyeah/pangu-350M](https://huggingface.co/sunzeyeah/pangu-350M) | [Pangu-350M](https://pan.baidu.com/s/1IzgtW48S2PKyjxPPMe1rAQ) |  c5jj |
| [Pangu-2.6B](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha)   | 9.8GB | [sunzeyeah/pangu-2_6B](https://huggingface.co/sunzeyeah/pangu-2_6B) | [Pangu-2.6B](https://pan.baidu.com/s/1Tzvja4q_LgQOwkWPQ4jShw)    | 2rad |
| [Pangu-13B](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha)   | 23.6GB | [sunzeyeah/pangu-13B](https://huggingface.co/sunzeyeah/pangu-13B) | [Pangu-13B](https://pan.baidu.com/s/11fWAeYYKqI7pH0UiuJ5jEQ)    | u3dx |
| [GLM-350M-chinese](https://github.com/THUDM/GLM) | 679MB | [sunzeyeah/glm-350M-chinese](https://huggingface.co/sunzeyeah/glm-350M-chinese) | [GLM-350M-chinese](https://pan.baidu.com/s/11Lef-E7Tsz5OGOueCpiqaA) | ii8e |
| [GLM-10B-chinese](https://github.com/THUDM/GLM)   | 18.4G | [sunzeyeah/glm-10B-chinese](https://huggingface.co/sunzeyeah/glm-10B-chinese) | [GLM-10B-chinese](https://pan.baidu.com/s/1GuOefx42n_GzFfwnjoBltw) | fynj  |
| [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)   | 25.6G | [sunzeyeah/chatglm-6B](https://huggingface.co/sunzeyeah/chatglm-6B) | [ChatGLM-6B](https://pan.baidu.com/s/1OlpkMeQD6-LEpNFWx5E-mg) | uq1k |

**PS**: 本repo提供的预训练模型下载中，
- 对于pytorch_model\*.bin
  - 如果源文件已包括，则不做改动
  - 如果源文件不包括，则根据其提供的checkpoint转换为pytorch_model\*.bin
- 其余文件可能相对原文件有改动，包括：modeling_\*.py、tokenization_\*.py、configuration_\*.py、config.json和tokenizer.config

### 2. 数据下载

| 数据集      | size | huggingface地址 | 百度网盘地址  |  提取码      | 
| ----------- | ----------- | ----------- |  ----------- |  ----------- |
| [CLUE Benchmark](https://github.com/CLUEbenchmark/CLUE)   | 500MB | | [CLUE Benchmark](https://pan.baidu.com/s/15F60nRbBd6d6UvyXdwbXQA) |  m6gt |
| SFT & Reward Data  | 5GB | [sunzeyeah/chinese_chatgpt_corpus](https://huggingface.co/datasets/sunzeyeah/chinese_chatgpt_corpus) | [SFT & Reward Data](https://pan.baidu.com/s/1sl8PB-Dlt1xLIYczMODyRg) | ecyc |
| [百科](https://github.com/brightmart/nlp_chinese_corpus)  | 652MB | | [baike_qa_2019](https://pan.baidu.com/s/1N6I-fvx6FLHwuxJuDLLA8g) | 7jad |
| [知道问答](https://github.com/SophonPlus/ChineseNlpCorpus) | 847MB | | [zhidao](https://pan.baidu.com/s/1sjR3vABiMbdV1HkUt6kCKQ) | neds |
| [对联](https://github.com/wb14123/couplet-dataset/releases/download/1.0/couplet.tar.gz)  | 221MB | | [couplets](https://pan.baidu.com/s/1oo6oaephPXpyU-sDd_37qg) | 54ey |
| [古文](https://github.com/NiuTrans/Classical-Modern)  | 125MB | | [Classical & Modern](https://pan.baidu.com/s/1ZyGkYOgzT9ZEVnpSpB4kQg) | a4cr |
| [古诗词](https://github.com/chinese-poetry/chinese-poetry)  | 87MB | | [chinese poetry](https://pan.baidu.com/s/13uvkA96PdKpKB7ZQ2GkXZQ) | 5zzj |
| 微博新闻评论  | 522MB | | [weibo summary comments](https://pan.baidu.com/s/1h45O0q6gQl3LbH-NjzuRlw) | w0g1 |

**PS**: SFT & Reward Data基于百科、知道问答、对联、古文、古诗词、微博新闻评论数据构造，可直接用于SFT和Reward阶段训练。详见[data_prepare.py](./src/data_prepare.py)


## Usage

### 1. LLM模型预训练
对开源LLM进行增量预训练，基于deepspeed实现。目前支持2类模型架构：
- decoder结构：LLaMA、Pangu
- encoder结构：GLM、ChatGLM

```bash
cd examples
bash pretrain.sh
```

### 2. LLM模型评测
对开源中文LLM进行ZeroShot、OneShot或FewShot的评测，评测任务和数据集使用[CLUEBenchmark](https://github.com/CLUEbenchmark/CLUE) ，评测方法和prompt模板参考[Pangu-alpha论文](https://arxiv.org/abs/2104.12369) 。详见[eval_pretrain.py](./src/eval_pretrain.py) 和 [data.py](src/data/data.py)

目前支持5个开源模型: 
- Pangu-350M
- Pangu-2.6B
- Pangu-13B
- GLM-350M-chinese
- GLM-10B-chinese

```bash
cd examples
bash eval_pretrain.sh
```

### 3. SFT
使用开源LLM + SFT&Reward数据进行SFT训练
```bash
cd examples
bash train_sft.sh
```
### 4. Reward Model
使用SFT模型 + SFT&Reward数据进行Reward模型训练。训练时，将SFT模型的前70%层固定，不进行梯度更新
```bash
cd examples
bash train_reward.sh
```

### 5. RLHF
利用PPO算法和Reward Model，进一步更新SFT模型。基于开源框架[DeepSpeedChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) 实现
```bash
cd examples
bash train_rlhf.sh
```


## Results

### 1. LLM模型评测

<details>
<summary><b>验证集(dev.json)结果</b></summary>

<table>
    <tr>  <td rowspan="2">Dataset</td>  <td rowspan="2">Method</td>  <td rowspan="2">Metrics</td>  <td rowspan="2">Task Type</td>  <td colspan="5" style="text-align:center">Zero-shot</td>  <td colspan="5" style="text-align:center">Few-shot</td> </tr>
    <tr>  <td>GLM-350M-chinese</td>  <td>Pangu-350M</td>  <td>Pangu-2.6B</td>  <td>GLM-10B-chinese</td>  <td>Pangu-13B</td>  <td>GLM-350M-chinese</td>  <td>Pangu-350M</td>  <td>Pangu-2.6B</td>  <td>GLM-10B-chinese</td>  <td>Pangu-13B</td> </tr>
    <tr>  <td>OCNLI</td>  <td>PPL</td>  <td>acc</td>  <td>NLI</td>  <td>0.3074</td>  <td style="color:red"><b>0.3369</b></td>  <td>0.3061</td>  <td>0.3288</td>  <td>0.3301</td>  <td>0.3298</td>  <td>0.3352</td>  <td>0.3216</td>  <td></td>  <td></td> </tr>
    <tr>  <td>CMNLI</td>  <td>PPL</td>  <td>acc</td>  <td>NLI</td>  <td>0.3279</td>  <td>0.3302</td>  <td>0.3310</td>  <td>0.3338</td>  <td style="color:red"><b>0.3358</b></td>  <td>0.3356</td>  <td>0.3328</td>  <td>0.3300</td>  <td></td>  <td></td> </tr>
    <tr>  <td>CHID</td>  <td>PPL</td>  <td>acc</td>  <td>Cloze(multi-choices)</td>  <td>0.0734</td>  <td>0.0916</td>  <td>0.0670</td>  <td>0.1016</td>  <td style="color:red"><b>0.1018</b></td>  <td>0.0979</td>  <td>0.1007</td>  <td>0.0996</td>  <td></td>  <td></td> </tr>
    <tr>  <td>CMRC2018</td>  <td>generation</td>  <td>f1</td>  <td>MRC</td>  <td>0.093</td>  <td>0.0979</td>  <td>0.1007</td>  <td style="color:red"><b>0.1392</b></td>  <td>0.021</td>  <td>0.09345</td>  <td>0.097</td>  <td>0.1007</td>  <td></td>  <td></td> </tr>
    <tr>  <td>CLUEWSC2020</td>  <td>PPL</td>  <td>acc</td>  <td>WSC</td>  <td>0.4934</td>  <td>0.5328</td>  <td style="color:red"><b>0.5592</b></td>  <td>0.5131</td>  <td>0.4671</td>  <td>0.5526</td>  <td>0.4473</td>  <td>0.4671</td>  <td></td>  <td></td> </tr>
    <tr>  <td>C3</td>  <td>PPL</td>  <td>acc</td>  <td>Common sense reasoning</td>  <td>0.2360</td>  <td>0.2426</td>  <td>0.2418</td>  <td style="color:red"><b>0.2573</b></td>  <td>0.2567</td>  <td>0.2476</td>  <td>0.2559</td>  <td>0.2515</td>  <td></td>  <td></td> </tr>
    <tr>  <td>AFQMC</td>  <td>PPL</td>  <td>acc</td>  <td>Text classification</td>  <td style="color:red"><b>0.6306</b></td>  <td>0.4582</td>  <td>0.4914</td>  <td>0.4960</td>  <td>0.5000</td>  <td>0.4872</td>  <td>0.4993</td>  <td>0.5018</td>  <td></td>  <td></td> </tr>
    <tr>  <td>CSL</td>  <td>PPL</td>  <td>acc</td>  <td>Text classification</td>  <td>0.4943</td>  <td>0.4913</td>  <td>0.4666</td>  <td style="color:red"><b>0.5126</b></td>  <td>0.4996</td>  <td>0.5140</td>  <td>0.5036</td>  <td>0.4973</td>  <td></td>  <td></td> </tr>
    <tr>  <td>IFLYTEK</td>  <td>PPL</td>  <td>acc</td>  <td>Text classification</td>  <td>0.1292</td>  <td style="color:red"><b>0.3058</b></td>  <td>0.265</td>  <td>0.2620</td>  <td>0.2408</td>  <td>0.2539</td>  <td>0.2535</td>  <td>0.2524</td>  <td></td>  <td></td> </tr>
    <tr>  <td>TNEWS</td>  <td>PPL</td>  <td>acc</td>  <td>Text classification</td>  <td>0.1582</td>  <td>0.2022</td>  <td>0.2449</td>  <td>0.2489</td>  <td style="color:red"><b>0.2527</b></td>  <td>0.2555</td>  <td>0.2466</td>  <td>0.2494</td>  <td></td>  <td></td> </tr>
</table>
</details>

### 2. SFT

模型训练参数：

| 模型 | 可训练参数量 | 数据量 | batch size | sequence length | 硬件 | 显存占用 | speed | Hours per epoch |
| --- | --- | --- | --- | :---: | :---: | :---: | --- | --- |
| GLM-350M-chinese | 355M | 5.4M | 4 | 512 | V100 16G | 13G | 3.7 s/iter | 88h
| Pangu-350M | 345M | 5.4M | 48 | 512 | A100 80G | 78G | 1.91 s/iter | 27.5h
| Pangu-2.6B | 2.6B | 5.4M |  8 | 512 | A100 80G | 79.4G | 9.61 s/iter | 116h

SFT模型下载：

| 预训练模型  |  SFT数据集  | size | huggingface地址 | 百度网盘地址  |  提取码      | 
| ----------- | --- | ----------- | ----------- |  ----------- | ----------- |
| Pangu-350M |  SFT & Reward Data | 1.3GB | [sunzeyeah/pangu-350M-sft](https://huggingface.co/sunzeyeah/pangu-350M-sft) | [Pangu-350M-SFT](https://pan.baidu.com/s/14nF63nAQz38jmplUC0EQ-g) | 0f8j |
| Pangu-2.6B |  SFT & Reward Data | 9.8GB | [sunzeyeah/pangu-2_6B-sft](https://huggingface.co/sunzeyeah/pangu-2_6B-sft) | [Pangu-2.6B-SFT](https://pan.baidu.com/s/1Q2PKf0MnFsiFlNhU-6WIrg) | r13w |


<details>
<summary><b>Pangu-2.6B-SFT生成示例</b></summary>

```
# 生成参数
max_length = 512
max_length_generation = 100
top_k = 10
temperature = 0.8
do_sample=True
```
```
[Input] 我不能确定对方是不是喜欢我,我却想分分秒秒跟他在一起,有谁能告诉我如何能想他少一点<sep>回答:
[Generated] 你爱他就多关心他,少在他面前表现出你的脆弱。这样他才更会爱你。


[Input] 一次次买彩,一次次倾听那比分,一次次兴奋,一次次失落,一次次无归而返!最终还是任8~~~~<sep>回答:
[Generated] 我也是这样,买彩的时间也不长,但是每次都是这样。。。。。。。

[Input] 现代文:太祖很高兴地说:凭着这样的部下去作战,还担心不能取胜么?<sep>古文:
[Generated] 太祖悦曰:此自是以去,何忧不胜?

[Input] 现代文:十一年秋,丞相伯颜与阿术会师襄阳,遣阿剌罕率诸翼军攻郢、复诸州,十月夺取郢州之南门堡。<sep>古文:
[Generated] 十一年秋,丞相伯颜与阿术会师襄阳,遣阿剌罕率诸翼军攻郢、复诸州,十月夺郢州之南门堡。

[Input] 以孔⁇的风格,写一篇经书,题为中庸<sep>
[Generated] 我闻昔者夫子之言也,曰:吾道一以贯之。其或行之或不行,吾未知之。是故君子慎其独也。吾未见好德如好色者也。好德如好色者,未之有也。好德如好色者,既已之有也。君子不以言举人,不以人废言。好德如好色者,其如命何!吾

[Input] 以陆游的风格,写一首七言诗,题为晨起<sep>
[Generated] 秋光满纸清⁇明,睡醒一江万马奔。自笑身如枯木鹤,何妨食饱作闲人。

[Input] 上联:霜风几笔雕残画<sep>下联:
[Generated] 云海千山叠翠峰

[Input] 新闻内容:据悉,这名14岁的女生在Facebook上发布了戏谑考试的博文,其中使用了胡志明1946年反对法国殖民者的讲话,随后被越南当局停课一年。越南官员称,该学生歪曲了历史事件,侮辱了老师的情感。多数网友虽对越南当局做法不满,但都表示,至少他们还能上Facebook.<sep>摘要:越南女生Facebook上模仿胡志明讲话被停课<sep>评论:
[Generated] 这女的,还真有勇气,我喜欢你!

[Input] 新闻内容:习近平天津考察时勉励大学生志存高远、脚踏实地,转变择业观念,坚持从实际出发,勇于到基层一线和艰苦地方去,把人生的路一步步走稳走实,善于在平凡岗位上创造不平凡的业绩<sep>摘要:习近平勉励大学生转变择业观念勇于到基层和艰苦地方去<sep>评论:
[Generated] 现在的社会,很多年轻人都是在父母的安排下,去了不同的单位,在不同的地方工作,在不同的城市生活,这就导致了年轻人的思维方式,价值观都不一样,这也导致了年轻人的成长道路,也有很多的不安全感,这也是很正常的事情,年轻人需要正确的引导,正确的引导,也需要正确的引导!

[Input] 问题:劳动仲裁申请法院强制执行,法院强制执行需要多长时间<sep>回答:
[Generated] 一般申请法院强制执行需要3-6个月时间,具体需要看法院的执行措施。

[Input] 问题:银行贷款50万,30年!年利率是4.41怎么算一个月还多少钱<sep>回答:
[Generated] 若申请的是招行贷款,要计算贷款利息或者每月还款金额,需要知道贷款本金,贷款期限,还款方式,贷款年利率。若上述信息都确认到,可以通过我行贷款计算器尝试计算,登录招行官网右下方找到“理财计算器”--“个人贷款计算器”可以计算。
```
</details>


### 3. Reward Model

模型训练参数：

| 模型 | 可训练参数量 | 数据量 | batch size | sequence length | 硬件 | 显存占用 | speed | Hours per epoch |
| --- | --- | --- | --- | :---: | :---: | :---: | --- | --- |
| Pangu-350M | 131M | 12M | 32 | 512 | A100 80G | 72.6G | 1.91 s/iter | 105h
| Pangu-2.6B | 815M | 12M |  8 | 512 | A100 80G | 80.7G |  | 423h


Reward模型下载：

| 预训练模型  |  SFT数据集  | size | huggingface地址 | 百度网盘地址  |  提取码      | 
| ----------- | --- | ----------- | ----------- |  ----------- | ----------- |
| Pangu-350M |  SFT & Reward Data | 1.3GB | [sunzeyeah/pangu-350M-reward](https://huggingface.co/sunzeyeah/pangu-350M-reward) | [Pangu-350M-Reward](https://pan.baidu.com/s/1wC3w78t7pVn0Xn5tJHy06A) | 4gju |


### 4. RLHF

To be updated

### 5. DeepSpeed实验

为验证不同预训练模型使用deepspeed的训练效率是否能达到官方宣称的效果（加速、节省GPU等），进行了benchmarking
- 实验场景：SFT阶段训练
- 实验参数：```max_sequence_length=512```

<details>
<summary><b>DeepSpeed实验结果</b></summary>
<table>
   <tr> <td>模型</td> <td>数据</td>  <td>整体耗时/epoch</td>  <td>单条样本耗时</td>  <td>内存使用量</td>  <td>显存使用量</td>  <td>GPU型号和数量</td> <td>fp16</td> <td>bf16</td> <td>deepspeed stage</td> <td>offload optimizer</td> <td>pin memory</td> <td>offloard param</td> <td>overlap comm</td> <td>allgather bucket size</td> <td>stage3 max live parameters</td> <td>batch size</td> <td>gradient accumulation steps</td> <td>gradient checkpointing</td> <td>model half</td> </tr>
   <tr> <td rowspan="11">T5-large</td> <td rowspan="11">wmt16-en-ro, 共计61万条样本</td> <td>43h</td>  <td>0.5s/it</td>  <td>7.1G</td>  <td>1*14529MB</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>-</td>  <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>-</td> <td>2</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td>152h</td>  <td>1.78s/it</td>  <td>38.26G</td>  <td>1*11663MB</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>2e8</td> <td>-</td> <td>2</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td>250h</td>  <td>2.95s/it</td>  <td>38.74G</td>  <td>1*7255MB</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>1e5</td> <td>-</td> <td>2</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td>62h</td>  <td>5.8s/it</td>  <td>86.81G</td>  <td>8*7811MB</td>  <td>8*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>1e5</td> <td>-</td> <td>2</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>2e8</td> <td>-</td> <td>16</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>1e5</td> <td>-</td> <td>16</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td>290h</td>  <td>3.48s/it</td>  <td>46.53G</td>  <td>1*6655MB</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>2e8</td> <td>2e8</td> <td>2</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td>380h</td>  <td>4.5s/it</td>  <td>43.48G</td>  <td>1*5263MB</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e5</td> <td>1e5</td> <td>2</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td>215h</td>  <td>4.9s/it</td>  <td>47.31G</td>  <td>2*5019MB</td>  <td>2*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e5</td> <td>1e5</td> <td>2</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td>1370h</td>  <td>64s/it</td>  <td>57.55G</td>  <td>4*4701MB</td>  <td>4*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e5</td> <td>1e5</td> <td>2</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td>948h</td>  <td>90s/it</td>  <td>72.54G</td>  <td>8*4585MB</td>  <td>8*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e5</td> <td>1e5</td> <td>2</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td rowspan="7">Pangu-2.6B</td> <td rowspan="7">SFT & Reward Data的验证集，共1万条样本</td> <td>2h</td>  <td>5.76s/it</td>  <td>67.86G</td>  <td>1*15631MB</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>2e8</td> <td>-</td> <td>2</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td>2.1h</td>  <td>6.15s/it</td>  <td>67.88G</td>  <td>1*15705MB</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>1e5</td> <td>-</td> <td>2</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td>4.5h</td>  <td>13.3s/it</td>  <td>81.02G</td>  <td>1*15449MB</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>2e8</td> <td>2e8</td> <td>2</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td>11.5h</td>  <td>8.2s/it</td>  <td>75.89G</td>  <td>1*15299MB</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e5</td> <td>1e5</td> <td>2</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td>5.5h</td>  <td>7.8s/it</td>  <td>81.16G</td>  <td>2*14851MB</td>  <td>2*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e5</td> <td>1e5</td> <td>2</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td>6.2h</td>  <td>18.3s/it</td>  <td>97.31G</td>  <td>4*14389MB</td>  <td>4*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e5</td> <td>1e5</td> <td>2</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td>6.6h</td>  <td>38s/it</td>  <td>118.82G</td>  <td>8*14335MB</td>  <td>8*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e5</td> <td>1e5</td> <td>2</td> <td>8</td> <td>false</td> <td>false</td> </tr>
   <tr> <td rowspan="14">ChatGLM-6B</td> <td rowspan="14">SFT & Reward Data的验证集，共1万条样本</td> <td>-</td>  <td>-</td>  <td>120.45G</td>  <td>OOM</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>1e5</td> <td>-</td> <td>1</td> <td>8</td> <td>true</td> <td>true</td> </tr>
   <tr> <td>-</td>  <td>-</td>  <td>120.48G</td>  <td>OOM</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>2</td>  <td>true</td> <td>true</td> <td>-</td> <td>false</td> <td>1e3</td> <td>-</td> <td>1</td> <td>8</td> <td>false</td> <td>true</td> </tr>
   <tr> <td>-</td>  <td>-</td>  <td>153.02G</td>  <td>OOM</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e2</td> <td>1e2</td> <td>1</td> <td>8</td> <td>false</td> <td>true</td> </tr>
   <tr> <td>-</td>  <td>-</td>  <td>154G</td>  <td>OOM</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>2e8</td> <td>2e8</td> <td>1</td> <td>8</td> <td>true</td> <td>true</td> </tr>
   <tr> <td>21.2h</td>  <td>60s/it</td>  <td>154G</td>  <td>1*10443MB</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>2e8</td> <td>auto</td> <td>1</td> <td>8</td> <td>true</td> <td>true</td> </tr>
   <tr> <td>21.5h</td>  <td>60s/it</td>  <td>152.81G</td>  <td>1*10409MB</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e5</td> <td>1e5</td> <td>1</td> <td>8</td> <td>true</td> <td>true</td> </tr>
   <tr> <td>23.5h</td>  <td>65s/it</td>  <td>153.36G</td>  <td>1*9229MB</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e3</td> <td>1e3</td> <td>1</td> <td>8</td> <td>true</td> <td>true</td> </tr>
   <tr> <td>14h</td>  <td>80s/it</td>  <td>158.21G</td>  <td>2*8631MB</td>  <td>2*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e3</td> <td>1e3</td> <td>1</td> <td>8</td> <td>true</td> <td>true</td> </tr>
   <tr> <td>7.8h</td>  <td>90s/it</td>  <td>168.38G</td>  <td>4*6743MB</td>  <td>4*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e3</td> <td>1e3</td> <td>1</td> <td>8</td> <td>true</td> <td>true</td> </tr>
   <tr> <td>4h</td>  <td>90s/it</td>  <td>189.34G</td>  <td>8*6729MB</td>  <td>8*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e3</td> <td>1e3</td> <td>1</td> <td>8</td> <td>true</td> <td>true</td> </tr>
   <tr> <td>1h</td>  <td>100s/it</td>  <td>189.38G</td>  <td>8*10047MB</td>  <td>8*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e3</td> <td>1e3</td> <td>4</td> <td>8</td> <td>true</td> <td>true</td> </tr>
   <tr> <td>50min</td>  <td>40s/it</td>  <td>189.39G</td>  <td>8*14763MB</td>  <td>8*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e3</td> <td>1e3</td> <td>8</td> <td>2</td> <td>true</td> <td>true</td> </tr>
   <tr> <td>35min</td>  <td>113s/it</td>  <td>189.39G</td>  <td>8*14763MB</td>  <td>8*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e3</td> <td>1e3</td> <td>8</td> <td>8</td> <td>true</td> <td>true</td> </tr>
   <tr> <td>-</td>  <td>-</td>  <td>189.34G</td>  <td>OOM</td>  <td>8*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e3</td> <td>1e3</td> <td>10</td> <td>8</td> <td>true</td> <td>true</td> </tr>
   <tr> <td rowspan="11">GLM-10B-Chinese</td> <td rowspan="11">SFT & Reward Data的验证集，共1万条样本</td> <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>2e8</td> <td>2e8</td> <td>1</td> <td>8</td> <td>true</td> <td>false</td> </tr>
   <tr> <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>2e8</td> <td>auto</td> <td>1</td> <td>8</td> <td>true</td> <td>false</td> </tr>
   <tr> <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e5</td> <td>1e5</td> <td>1</td> <td>8</td> <td>true</td> <td>false</td> </tr>
   <tr> <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e3</td> <td>1e3</td> <td>1</td> <td>8</td> <td>true</td> <td>false</td> </tr>
   <tr> <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>1*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e2</td> <td>1e2</td> <td>1</td> <td>8</td> <td>true</td> <td>false</td> </tr>
   <tr> <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>2*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e2</td> <td>1e2</td> <td>1</td> <td>8</td> <td>true</td> <td>false</td> </tr>
   <tr> <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>4*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e2</td> <td>1e2</td> <td>1</td> <td>8</td> <td>true</td> <td>false</td> </tr>
   <tr> <td>-</td>  <td>-</td>  <td>OOM</td>  <td>-</td>  <td>8*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e2</td> <td>1e2</td> <td>1</td> <td>8</td> <td>true</td> <td>false</td> </tr>
   <tr> <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>4*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e2</td> <td>1e2</td> <td>1</td> <td>8</td> <td>true</td> <td>true</td> </tr>
   <tr> <td>-</td>  <td>-</td>  <td>-</td>  <td>OOM</td>  <td>6*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e2</td> <td>1e2</td> <td>1</td> <td>8</td> <td>true</td> <td>true</td> </tr>
   <tr> <td>-</td>  <td>-</td>  <td>OOM</td>  <td>-</td>  <td>8*V100 16G</td>  <td>true</td>  <td>-</td>  <td>3</td>  <td>true</td> <td>true</td> <td>true</td> <td>false</td> <td>1e2</td> <td>1e2</td> <td>1</td> <td>8</td> <td>true</td> <td>true</td> </tr>
</table>
</details>

**PS**: deepspeed的参数介绍和调优经验，可参见[DeepSpeed Configuration](src/resources/config/deepspeed/README.md)

### 6. LoRA实验

为验证LoRA的训练效率提升，进行了benchmarking

- 实验场景：SFT阶段训练
- 实验数据：SFT & Reward Data的验证集，共1万条样本
- 实验参数：```max_sequence_length=512, lora_alpha=1, lora_train_bias='none'```

<details>
<summary><b>LoRA实验结果</b></summary>
<table>
   <tr> <td>模型</td> <td>LoRA rank</td> <td>可训练参数量</td> <td>deepspeed</td> <td>batch size</td> <td>GPU型号和数量</td> <td>显存使用量</td> <td>单条样本耗时</td> <td>整体耗时/epoch</td> </tr>
   <tr> <td rowspan="8">Pangu-2.6B</td>  <td>-</td>  <td>2.6B</td>  <td>-</td>  <td>8</td>  <td>1*A100 80G</td>  <td>1*79421MB</td>  <td>9.66s/it</td>  <td>12.5min</td> </tr>
   <tr> <td>1000</td>  <td>1.5B</td>  <td>-</td>  <td>8</td>  <td>1*A100 80G</td>  <td>1*76129MB</td>  <td>11.61s/it</td>  <td>15min</td> </tr>
   <tr> <td>500</td>  <td>758MB</td>  <td>-</td>  <td>12</td>  <td>1*A100 80G</td>  <td>1*77179MB</td>  <td>16.2s/it</td>  <td>14min</td> </tr>
   <tr> <td>100</td>  <td>151MB</td>  <td>-</td>  <td>16</td>  <td>1*A100 80G</td>  <td>1*81103MB</td>  <td>18.6s/it</td>  <td>12min</td> </tr>
   <tr> <td>50</td>  <td>75MB</td>  <td>-</td>  <td>16</td>  <td>1*A100 80G</td>  <td>1*80809MB</td>  <td>17.8s/it</td>  <td>11.5min</td> </tr>
   <tr> <td>10</td>  <td>15MB</td>  <td>-</td>  <td>16</td>  <td>1*A100 80G</td>  <td>1*78735MB</td>  <td>17.6s/it</td>  <td>11.5min</td> </tr>
   <tr> <td>100</td>  <td>151MB</td>  <td>stage=2, w offloading</td>  <td>24</td>  <td>1*A100 80G</td>  <td>1*76933MB</td>  <td>25.5s/it</td>  <td>11min</td> </tr>
   <tr> <td>100</td>  <td>151MB</td>  <td>stage=3, w offloading</td>  <td>24</td>  <td>1*A100 80G</td>  <td>1*77259MB</td>  <td>46.5s/it</td>  <td>20min</td> </tr>
   <tr> <td rowspan="3">ChatGLM-6B</td>  <td>-</td>  <td>6.2B</td>  <td>-</td>  <td>3</td>  <td>1*A100 80G</td>  <td>1*79206MB</td>  <td>6.7s/it</td>  <td>23.5min</td> </tr>
   <tr> <td>1000</td>  <td>1.9B</td>  <td>-</td>  <td>6</td>  <td>1*A100 80G</td>  <td>1*78840MB</td>  <td>12.8s/it</td>  <td>22.5min</td> </tr>
   <tr> <td>500</td>  <td>994MB</td>  <td>-</td>  <td>6</td>  <td>1*A100 80G</td>  <td>1*68832MB</td>  <td>12.4s/it</td>  <td>21.5min</td> </tr>
</table>
</details>