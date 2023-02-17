<<<<<<< HEAD
## 使用RLHF训练Pangu 2.6B中文对话模型
=======
## 基于`trlx`库使用RLHF训练Pangu 2.6B中文对话模型pipeline
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1

我们的pipeline是基于OpenAI论文 "[Learning to Summarize from human feedback](https://arxiv.org/abs/2009.01325)"的开源代码进行修改。


### 准备阶段

<<<<<<< HEAD
a).  需要配置trlx库相关环境，参考 "[trlx] (https://github.com/CarperAI/trlx)"
=======
1).  需要配置trlx库相关环境，参考 "[trlx] (https://github.com/CarperAI/trlx)"
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1

```bash
git clone https://github.com/CarperAI/trlx.git
cd trlx
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116 # for cuda
pip install -e .
```

<<<<<<< HEAD
b).  准备webtext数据集: 

```bash
    https://paperswithcode.com/dataset/webtext
```
保存至: ./dialogue_dir

c).  下载盘古-2.6B模型: 

```bash
    https://huggingface.co/imone/pangu_2_6B
```

d).  基于模型的输出收集人工反馈数据集（待开源）
保存至: ./reward_data_dir
=======
2).  下载盘古-2.6B模型: 

```bash
    https://huggingface.co/imone/pangu_2_6B
```

3).  准备SFT数据集(以webtext为例): 

```bash
    https://paperswithcode.com/dataset/webtext
```
保存至: ./dialogue_dir/demo.json

4).  收集人工反馈数据
保存至: ./reward_data_dir/processed/demo.json
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1


### 训练步骤

1). 监督微调 (SFT):

    cd sft/ && deepspeed train_SFT.py

<<<<<<< HEAD
2). 使用人工反馈数据集训练 Reward 模型:
=======
2). 训练 Reward 模型:
>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1

    cd reward_model/ && deepspeed train_reward_model.py

3). 使用PPO算法强化学习:

    accelerate launch --config_file configs/default_accelerate_config.yaml trlx_pangu_rlhf.py

   备注: 至少需要1张V100显卡。

## 参考文献

1. Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano, "[Learning to Summarize from human feedback](https://arxiv.org/abs/2009.01325)", Neural Information Processing Systems, 2020.
