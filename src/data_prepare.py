
import sys
sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")
sys.path.insert(0, "/mnt/sfevol775196/sunzeye273/Code/chatgpt")
# sys.path.insert(0, "/mnt/share-pa002-vol682688-prd/sunzeye273/Code/chatgpt")
sys.path.insert(0, "/mnt/pa002-28359-vol543625-private/Code/chatgpt")
import os
import time
import json
import argparse
import random
import glob
import pandas as pd
import opencc

from tqdm import tqdm
from transformers import AutoTokenizer

from src.utils.logger import logger
from src.utils.file_utils import set_seed
from src.utils.nlp_utils import clean_text


def weibo_summary_comment(args, tokenizer):
    ''' 微博新闻+摘要+评论，每条评论有点赞数，可以根据点赞数构造reward训练集

    :param args:
    :param tokenizer:

    :return: processed json list
    '''
    t = time.time()
    fi = os.path.join(args.data_dir, "weibo_summary_comments_json.json")
    fo = os.path.join(args.output_dir, "weibo_summary_comments.jsonl")
    data = []
    with open(fo, "w", encoding="utf-8") as w:
        with open(fi, "r", encoding="utf-8") as r:
            while True:
                line = r.readline()
                if not line:
                    break

                item = json.loads(line.strip("\n"))
                article = item['article'].replace(" ", "")
                abstract = item['abstract'].replace(" ", "")
                prompt = f"新闻内容：{article} 摘要：{abstract}"
                prefix = "评论："
                answers = [
                    {
                        "answer": k.replace(" ", ""),
                        "score": int(v)
                    } for (k, v) in sorted(item['comments'], key=lambda x: (int(x[1]), len(x[0])), reverse=True)
                ]
                w.write(json.dumps({"prompt": prompt, "answers": answers}, ensure_ascii=False)+'\n')
                data.append({"prompt": prompt, "answers": answers, "prefix": prefix})
    logger.info(f"length: {len(data)}, time taken: {time.time()-t} s")

    return data


def couplets(args, tokenizer):
    ''' 对联数据（上联和下联），可以根据正确下联和负例下联，构造reward训练集

    :param args:
    :param tokenizer:

    :return: processed json list
    '''
    t1 = time.time()
    fi = os.path.join(args.data_dir, "couplets.txt")
    fo = os.path.join(args.output_dir, "couplets.jsonl")
    l2 = []
    nexts = dict()
    with open(fi, "r", encoding="utf-8") as r:
        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip("\n")
            idx = len(line) // 2
            prompt = line[:idx]
            answer = line[idx+1:]
            prefix = "下联："
            answers = [{"answer": answer, "score": 1}]
            l2.append({"prompt": f"上联：{prompt}", "answers": answers, "prefix": prefix})
            length = len(answer)
            if length not in nexts:
                nexts[length] = list()
            nexts[length].append(answer)
    t2 = time.time()
    logger.info(f"length: {len(l2)}, # different lengths: {len(nexts)}, time taken: {t2-t1} s")
    data = []
    with open(fo, "w", encoding="utf-8") as w:
        for l in tqdm(enumerate(l2), desc="Processing Couplets"):
            answer = l['answers'][0]
            length = len(answer['answer'])
            # 上下联长度一样
            nexts_tmp = set(nexts[length])
            nexts_tmp.remove(answer['answer'])
            nexts_tmp = set(nexts[length]).difference(set([answer['answer']]))
            #         nexts_tmp.remove(answer['answer'])
            answers.extend([{"answer": fa, "score": 0} for fa in random.sample(nexts_tmp, 2)])
            # 上下联长度不一样
            keys = set(nexts.keys())
            keys.remove(length)
            answers.extend([{"answer": random.choice(nexts[key]), "score": -1} for key in random.sample(keys, 2)])
            #         answers = sorted(answers, key=lambda x: x['score'], reverse=True)
            w.write(json.dumps({"prompt": l['prompt'], "answers": answers, "prefix": l['prefix']}, ensure_ascii=False)+'\n')
            data.append({"prompt": l['prompt'], "answers": answers, "prefix": l['prefix']})
    #         if i % 1000 == 0:
    #             logger.info(f"{i} samples processed, time taken: {time.time()-t2} s")
    logger.info(f"length: {len(data)}, time taken: {time.time()-t2} s")

    return data


def zhidao(args, tokenizer):
    ''' 百度知道的问答数据，每条问题有多个答案以及最佳答案，可以直接构造reward训练集

    :param args:
    :param tokenizer:

    :return: processed json list
    '''
    t = time.time()
    fp = os.path.join(args.data_dir, "zhidao", "*.csv")
    fo = os.path.join(args.output_dir, "zhidao.jsonl")
    data = []
    with open(fo, "w", encoding="utf-8") as w:
        for fi in glob.glob(fp):
            df = pd.read_csv(fi).sort_values(by=["title", "is_best"], ascending=False)
            prev_title = None
            prev_prompt = None
            prefix = "答："
            for _, val in df.iterrows():
                if isinstance(val['question'], str) and val['question'] != val['title']:
                    prompt = f"问题：{val['title']} 内容：{val['question']}"
                else:
                    prompt = f"问题：{val['title']}"
                if prev_title is not None and prev_title == val['title']:
                    answers.append({"answer": val['reply'], "score": val['is_best']})
                else:
                    if prev_title is not None:
                        #                     l3.append({"prompt": prev_prompt, "answers": copy.deepcopy(answers)})
                        w.write(json.dumps({"prompt": prev_prompt, "answers": answers}, ensure_ascii=False)+'\n')
                        data.append({"prompt": prev_prompt, "answers": answers})
                    answers = [{"answer": val['reply'], "score": val['is_best']}]
                prev_prompt = prompt
                prev_title = val['title']
            #         l3.append({"prompt": prev_prompt, "answers": copy.deepcopy(answers)})
            w.write(json.dumps({"prompt": prev_prompt, "answers": answers, "prefix": prefix}, ensure_ascii=False)+'\n')
            data.append({"prompt": prev_prompt, "answers": answers, "prefix": prefix})
            logger.info(f"finished processing {os.path.basename(fi)}")
    logger.info(f"length: {len(data)}, time taken: {time.time()-t} s")

    return data


def chinese_classical(args, tokenizer):
    ''' 文言文和现代文的对照翻译，每条文言文有对应的现代文翻译，可以根据正确翻译或原文+负例翻译或原文，构造reward训练集

    :param args:
    :param tokenizer:

    :return: processed json list
    '''
    t1 = time.time()
    fp = os.path.join(args.data_dir, "Classical-Modern", "bitext", "*")
    fo = os.path.join(args.output_dir, "chinese_classical.jsonl")
    l3 = []
    dicts = dict()
    for fi in glob.glob(fp):
        name = os.path.basename(fi)
        dicts[name] = {"古文": [], "现代文": []}
        with open(fi, "r", encoding="utf-8") as r:
            for i, line in enumerate(r):
                line = line.strip("\n")
                if line.startswith("古文"):
                    p1 = line[3:]
                    dicts[name]['古文'].append(p1)
                elif line.startswith("现代文"):
                    p2 = line[4:]
                    dicts[name]['现代文'].append(p2)
                elif p1 is not None and p2 is not None:
                    pair = [("古文", p1), ("现代文", p2)]
                    random.shuffle(pair)
                    prompt = f"{pair[0][0]}：{pair[0][1]}"
                    prefix = f"{pair[1][0]}："
                    answers = [{"answer": pair[1][1], "score": 1}]
                    l3.append({"prompt": prompt, "answers": answers, "prefix": prefix, "name": name})
                    p1 = None
                    p2 = None
    t2 = time.time()
    logger.info(f"length: {len(l3)}, # different names: {len(dicts)}, time taken: {t2-t1} s")
    data = []
    with open(fo, "w", encoding="utf-8") as w:
        for l in tqdm(enumerate(l3), desc="Processing Chinese Classical-Modern"):
            name = l['name']
            prompt = l['prompt']
            prefix = l['prefix']
            answer = l['answers'][0]['answer']
            if prompt.startswith("古文"):
                answer_type = '现代文'
            else:
                answer_type = '古文'
            samples_tmp = set(dicts[name][answer_type])
            samples_tmp.remove(answer)
            answers.extend([{"answer": fa, "score": 0} for fa in random.sample(samples_tmp, 2)])
            keys = set(dicts.keys())
            keys.remove(name)
            answers.extend([{"answer": random.choice(dicts[key][answer_type]), "score": -1} for key in random.sample(keys, 2)])
            w.write(json.dumps({"prompt": prefix, "answers": answers}, ensure_ascii=False)+'\n')
            data.append({"prompt": prefix, "answers": answers})
    #         if i % 100 == 0:
    #             logger.info(f"{i} samples processed, time taken: {time.time()-t2} s")
    logger.info(f"length: {len(data)}, time taken: {time.time()-t2} s")

    return data


def chinese_poetry(args, tokenizer):
    ''' 四书五经、诗、词、曲等古文数据，每篇文章或每首诗有作者、题目和正文，可以根据正确体裁/作者+负例体裁/作者，构造reward训练集

    :param args:
    :param tokenizer:

    :return: processed json list
    '''
    converter = opencc.OpenCC('t2s.json')
    t1 = time.time()
    fp = [
        # 四书五经
        os.path.join(args.data_dir, "chinese-poetry", "lunyu", "lunyu.json"),
        os.path.join(args.data_dir, "chinese-poetry", "sishuwujing", "*.json"),
        # 古体诗
        os.path.join(args.data_dir, "chinese-poetry", "caocaoshiji", "caocao.json"),
        os.path.join(args.data_dir, "chinese-poetry", "shijing", "shijing.json"),
        # 楚辞
        os.path.join(args.data_dir, "chinese-poetry", "chuci", "chuci.json"),
        # 诗
        os.path.join(args.data_dir, "chinese-poetry", "shi", "poet*.json"),
        # 词
        os.path.join(args.data_dir, "chinese-poetry", "ci", "ci*.json"),
        os.path.join(args.data_dir, "chinese-poetry", "nalanxingde", "*.json"),
        os.path.join(args.data_dir, "chinese-poetry", "wudai", "huajianji", "*juan.json"),
        os.path.join(args.data_dir, "chinese-poetry", "wudai", "nantang", "poetrys.json"),
        # 曲
        os.path.join(args.data_dir, "chinese-poetry", "yuanqu", "yuanqu.json"),
    ]
    fs = [each for f in fp for each in glob.glob(f)]
    
    l5 = []
    dicts = dict()
    for fi in fs:
        lines = json.load(open(fi, "r", encoding="utf-8"))
        if isinstance(lines, dict):
            lines = [lines]
        for i, line in enumerate(lines):
            if "lunyu" in fi:
                author = "孔子"
                genre = "经书"
                title = line['chapter']
                contents = "".join(line['paragraphs'])
            elif "daxue" in fi:
                author = "曾子"
                genre = "经书"
                title = "大学"
                contents = converter.convert("".join(line['paragraphs'])).replace("「", "“").replace("」", "”")
            elif "mengzi" in fi:
                author = "孟子"
                genre = "经书"
                title = converter.convert(line['chapter'])
                contents = converter.convert("".join(line['paragraphs'])).replace("「", "“").replace("」", "”")
            elif "zhongyong" in fi:
                author = "孔伋"
                genre = "经书"
                title = "中庸"
                contents = converter.convert("".join(line['paragraphs'])).replace("「", "“").replace("」", "”")
            elif "caocao" in fi:
                author = "曹操"
                genre = "古体诗"
                title = line['title']
                contents = "".join(line['paragraphs'])
            elif "shijing" in fi:
                author = "诗经"
                genre = "古体诗"
                title = line['chapter'] + "-" + line['section'] + "-" + line['title']
                contents = "".join(line['content'])
            elif "chuci" in fi:
                author = line['author']
                genre = "楚辞"
                title = line['section'] + "-" + line['title']
                contents = "".join(line['content'])
            elif "nalanxingde" in fi:
                author = line['author']
                genre = "词"
                title = line['title']
                contents = "".join(line['para'])
            elif "huajianci" in fi:
                author = line['author']
                genre = "词"
                title = line['title']
                contents = "".join(line['paragraphs'])
            elif "nantang" in fi:
                author = line['author']
                genre = "词"
                title = line['title']
                contents = "".join(line['paragraphs'])
            elif "yuanqu" in fi:
                author = line['author']
                genre = "曲"
                title = line['title']
                contents = "".join(line['paragraphs'])
            elif "shi" in fi:
                if len(line['paragraphs']) <= 0:
                    continue
                author = converter.convert(line['author'])
                genre = "五言诗" if len(line['paragraphs'][0]) == 12 else "七言诗"
                title = converter.convert(line['title'])
                contents = converter.convert("".join(line['paragraphs']))
            elif "ci" in fi:
                author = line['author']
                genre = "词"
                title = line['rhythmic']
                contents = "".join(line['paragraphs'])
            if genre not in dicts:
                dicts[genre] = dict()
            if author not in dicts[genre]:
                dicts[genre][author] = dict()
            quantifier = "篇" if genre in ["经书", "楚辞"] else "首"
            prompt = f"以{author}的风格，写一{quantifier}{genre}，题为{title}"
            answers = [{"answer": contents, "score": 1}]
            l5.append({"prompt": prompt, "answers": answers, "genre": genre, "title": title, "author": author})
            dicts[genre][author][title] = contents
    t2 = time.time()
    logger.info(f"length: {len(l5)}, # different lengths: {len(dicts)}, time taken: {t2-t1} s")
    data = []
    fo = os.path.join(args.output_dir, "chinese_poetry.jsonl")
    with open(fo, "w", encoding="utf-8") as w:
        for l in tqdm(enumerate(l5), desc="Processing Chinese Poetry"):
            genre = l['genre']
            author = l['author']
            title = l['title']
            prompt = l['prompt']
            answers = l['answers']
            # 同作者其他作品-2
            titles_tmp = set(dicts[genre][author].keys())
            titles_tmp.remove(title)
            if len(titles_tmp) > 0:
                t = random.choice(list(titles_tmp))
                answers.append({"answer": dicts[genre][author][t], "score": 0})
            # 同体裁其他作者其他作品-1
            authors_tmp = set(dicts[genre].keys())
            authors_tmp.remove(author)
            a = random.choice(list(authors_tmp))
            t = random.choice(list(dicts[genre][a].keys()))
            answers.append({"answer": dicts[genre][a][t], "score": -1})
            # 不同体裁作品-0
            genres_tmp = set(dicts.keys())
            genres_tmp.remove(genre)
            g = random.choice(list(genres_tmp))
            a = random.choice(list(dicts[g].keys()))
            t = random.choice(list(dicts[g][a].keys()))
            answers.append({"answer": dicts[g][a][t], "score": -2})
            w.write(json.dumps({"prompt": prompt, "answers": answers, "prefix": ""}, ensure_ascii=False)+'\n')
            data.append({"prompt": prompt, "answers": answers, "prefix": ""})
    logger.info(f"length: {len(data)}, time taken: {time.time()-t2} s")

    return data
    

def baike_qa_2019(args, tokenizer):
    ''' 百科问答数据集，每个问题只有一个答案，可构造sft训练集

    :param args:
    :param tokenizer:

    :return: processed json list
    '''
    fs = glob.glob(os.path.join(args.data_dir, "baike_qa2019", "baike_qa_*.json"))
    fo = os.path.join(args.output_dir, "baike_qa.jsonl")
    data = []
    t = time.time()
    with open(fo, "w", encoding="utf-8") as w:
        for f in fs:
            with open(f, "r", encoding="utf-8") as r:
                while True:
                    line = r.readline()
                    if not line:
                        break
                    item = json.loads(line.strip("\n"))
                    question = clean_text(item['title'] if len(item['title']) > len(item['desc']) else item['desc'])
                    prompt = question
                    prefix = "答："
                    answer = clean_text(item['answer'])
                    answers = [{"answer": answer, "score": 1}]
                    w.write(json.dumps({"prompt": prompt, "answers": answers, "prefix": prefix}, ensure_ascii=False)+'\n')
                    data.append({"prompt": prompt, "answers": answers, "prefix": prefix})
    logger.info(f"length: {len(data)}, time taken: {time.time()-t} s")

    return data


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return args


def main():
    args = get_parser()
    logger.info(f"Parameters: {args}")
    # set random seed
    set_seed(args.seed)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)

    # process raw datasets
    data_weibo = weibo_summary_comment(args, tokenizer)
    data_couplets = couplets(args, tokenizer)
    data_zhidao = zhidao(args, tokenizer)
    data_chinese_classical = chinese_classical(args, tokenizer)
    data_chinese_poetry = chinese_poetry(args, tokenizer)
    data_baike = baike_qa_2019(args, tokenizer)

    # merge processed datasets
    data = data_weibo + data_couplets + data_zhidao + data_chinese_classical + data_chinese_poetry + data_baike
    random.shuffle(data)
    fo = os.path.join(args.output_dir, "train_data_external_v1.jsonl")
    with open(fo, "w", encoding="utf-8") as w:
        for d in data:
            w.write(json.dumps(d, ensure_ascii=False)+'\n')

    fo = os.path.join(args.output_dir, "dev_data_external_v1.jsonl")
    with open(fo, "w", encoding="utf-8") as w:
        for d in data[:10000]:
            w.write(json.dumps(d, ensure_ascii=False)+'\n')

    logger.info("Finished saving processed train & dev files")


if __name__ == "__main__":
    main()
