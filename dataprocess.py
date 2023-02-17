<<<<<<< HEAD
import random
import os
import json
from copy import deepcopy
import jieba

def conv_gen(item_list):
    sample_list = []
    for item in item_list:
        conversation = item['conversation']
        prompt = ''
        for i, speak in enumerate(conversation):
            if speak['speaker'] == 'human':
                prompt = prompt + speak['content_info'][0]['query'] + " "
            if speak['speaker'] == 'bot':
                chosen = speak['best_response']
                for item2 in speak['content_info']:
                    if item2['query'] != chosen:
                        rejected = item2['query']
                        comment = item2['comment']
                        sample={}
                        sample['prompt'] = deepcopy(prompt).replace("\n","")
                        sample['prompt'] = sample['prompt'] + "模型回答："
                        sample['chosen'] = deepcopy(chosen).replace("\n","")
                        sample['rejected'] = deepcopy(rejected).replace("\n","")
                        # sample['prompt'] = " ".join(list(jieba.cut(sample['prompt'])))
                        # sample['chosen'] = " ".join(list(jieba.cut(sample['chosen'])))
                        # sample['rejected'] = " ".join(list(jieba.cut(sample['rejected'])))
                        print(sample)
                        sample_list.append(deepcopy(sample))

                prompt = prompt + chosen  + " "
    return sample_list

def process(path, seed=42):
    random.seed(seed)
    json_list = [f for f in os.listdir(path) if f.endswith(".json")]
    type2data = {f[:-5]: [] for f in json_list}
    for f_name in json_list:
        type = f_name[:-5]
        with open(f"{path}/{f_name}", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                type2data[type].append(item)

    train_part = []
    valid_part = []
    test_part = []

    for type in type2data:
        # N = len(type2data[type])
        for item in type2data[type]:
            p = random.random()
            if p < 0.8:
                train_part.append(item)
            else:
                test_part.append(item)

    train_list = conv_gen(train_part)
    test_part = conv_gen(test_part)

    return train_list,test_part

if __name__ == '__main__':
    cur_dir = "D:/work/Research_HUB/RLHF/OurData/merge"
    train,test =  process(cur_dir, seed=42)
    import torch
    torch.save(train, f"D:/work/Research_HUB/RLHF/trlx/examples/dialogue_rlhf/reward_data_dir/processed/train_data.pt")
    torch.save(test, f"D:/work/Research_HUB/RLHF/trlx/examples/dialogue_rlhf/reward_data_dir/processed/test_data.pt")

    for k in range(10):
        print(f"reward data demo-{k}:\n {train[-k-1]}")



=======
import random
import os
import json
from copy import deepcopy
import jieba

def conv_gen(item_list):
    sample_list = []
    for item in item_list:
        conversation = item['conversation']
        prompt = ''
        for i, speak in enumerate(conversation):
            if speak['speaker'] == 'human':
                prompt = prompt + speak['content_info'][0]['query'] + " "
            if speak['speaker'] == 'bot':
                chosen = speak['best_response']
                for item2 in speak['content_info']:
                    if item2['query'] != chosen:
                        rejected = item2['query']
                        comment = item2['comment']
                        sample={}
                        sample['prompt'] = deepcopy(prompt).replace("\n","")
                        sample['prompt'] = sample['prompt'] + "模型回答："
                        sample['chosen'] = deepcopy(chosen).replace("\n","")
                        sample['rejected'] = deepcopy(rejected).replace("\n","")
                        # sample['prompt'] = " ".join(list(jieba.cut(sample['prompt'])))
                        # sample['chosen'] = " ".join(list(jieba.cut(sample['chosen'])))
                        # sample['rejected'] = " ".join(list(jieba.cut(sample['rejected'])))
                        print(sample)
                        sample_list.append(deepcopy(sample))

                prompt = prompt + chosen  + " "
    return sample_list

def process(path, seed=42):
    random.seed(seed)
    json_list = [f for f in os.listdir(path) if f.endswith(".json")]
    type2data = {f[:-5]: [] for f in json_list}
    for f_name in json_list:
        type = f_name[:-5]
        with open(f"{path}/{f_name}", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                type2data[type].append(item)

    train_part = []
    valid_part = []
    test_part = []

    for type in type2data:
        # N = len(type2data[type])
        for item in type2data[type]:
            p = random.random()
            if p < 0.8:
                train_part.append(item)
            else:
                test_part.append(item)

    train_list = conv_gen(train_part)
    test_part = conv_gen(test_part)

    return train_list,test_part

if __name__ == '__main__':
    cur_dir = "D:/work/Research_HUB/RLHF/OurData/merge"
    train,test =  process(cur_dir, seed=42)
    import torch
    torch.save(train, f"D:/work/Research_HUB/RLHF/trlx/examples/dialogue_rlhf/reward_data_dir/processed/train_data.pt")
    torch.save(test, f"D:/work/Research_HUB/RLHF/trlx/examples/dialogue_rlhf/reward_data_dir/processed/test_data.pt")

    for k in range(10):
        print(f"reward data demo-{k}:\n {train[-k-1]}")



>>>>>>> 8d9fb60e1075f1df94c7bdd1fd1d8bb2c4440bb1
