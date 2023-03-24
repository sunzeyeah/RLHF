import collections
import sys
sys.path.insert(0, "/root/autodl-tmp/Code/RLHF")
sys.path.insert(0, "/mnt/private-pa002-vol726121-prd/Code/RLHF")
sys.path.insert(0, "/mnt/pa002-28359-vol543625-private/Code/RLHF")
import os
import argparse
import json
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torchmetrics.text.perplexity import Perplexity
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TextGenerationPipeline
)

from src.utils import logger
from src.data.data import (
    OCNLIDataset,
    CMNLIDataset,
    CHIDDataset,
    CMRCDataset,
    CLUEWSCDataset,
    C3Dataset,
    AFQMCDataset,
    CSLDataset,
    IFLYTEKDataset,
    TNEWSDataset
)
from src.utils.file_utils import set_seed


DATASET = {
    # NLI
    "ocnli": OCNLIDataset,
    "cmnli": CMNLIDataset,
    # Cloze and completion
    "chid": CHIDDataset,
    # MRC
    "cmrc2018": CMRCDataset,
    # Winograd
    "cluewsc2020": CLUEWSCDataset,
    # common sense reasoning
    "c3": C3Dataset,
    # Text Classification
    "tnews": TNEWSDataset,
    "iflytek": IFLYTEKDataset,
    "afqmc": AFQMCDataset,
    "csl": CSLDataset
}


def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_length_generation", type=int, default=100, help="Maximum number of newly generated tokens")

    # eval
    parser.add_argument("--eval_filename", type=str, default=None)
    parser.add_argument("--train_filename", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--max_few_shot", type=int, default=15, help="Maximum number of examples in few-shot evaulation")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.8)

    args = parser.parse_args()
    
    return args


def main():
    args = get_parser()
    logger.info(f"Parameters: {args}")

    set_seed(args.seed)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
    if "pangu" in args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
        model.resize_token_embeddings(tokenizer.vocab_size)
        # model.config.end_token_id = tokenizer.eos_token_id
        # model.config.pad_token_id = tokenizer.pad_token_id
        # model.config.bos_token_id = tokenizer.bos_token_id
        # model.config.eos_token_id = tokenizer.eos_token_id
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    logger.info(f"Finished loading model and tokenizer")

    # Set up the datasets
    dataset = DATASET.get(args.task, None)
    if dataset is None:
        raise ValueError(f"Unsupported task: {args.task}")
    train_filename = os.path.join(args.data_dir, args.train_filename) if args.train_filename is not None else None
    dev_dataset = dataset(args, os.path.join(args.data_dir, args.eval_filename),
                          tokenizer, train_filename)

    # Set up the metric
    perplexity = Perplexity(ignore_index=tokenizer.pad_token_id)

    def preprocess_logits_for_metrics(logits, labels):
        labels = labels.detach().cpu()
        probs = torch.softmax(logits, dim=-1).detach().cpu().to(torch.float32)
        ppls = []
        for i in range(probs.shape[0]):
            ppl = perplexity(probs[i:i+1], labels[i:i+1])
            ppls.append(ppl)

        return torch.stack(ppls)

    def calculate_f1(pred_text, label_text):
        pred_tokens = tokenizer(pred_text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False, return_tensors="pt")['input_ids'][0].tolist()
        label_tokens = tokenizer(label_text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False, return_tensors="pt")['input_ids'][0].tolist()
        common = collections.Counter(pred_tokens) & collections.Counter(label_tokens)
        num_same = sum(common.values())
        if len(pred_tokens) == 0 or len(label_tokens) == 0:
            return int(pred_tokens == label_tokens)
        if num_same == 0:
            return 0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(label_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    if args.train_filename is None:
        output_filename = os.path.join(args.output_dir, f"{args.task}_zeroshot_eval_result.jsonl")
    else:
        output_filename = os.path.join(args.output_dir, f"{args.task}_fewshot_eval_result.jsonl")

    if args.task in ["cmrc2018"]:
        # text_generator = TextGenerationPipeline(model, tokenizer, device=device)
        ems = []
        f1s = []
        with open(output_filename, "w", encoding="utf-8") as w:
            with torch.no_grad():
                for dev_data in tqdm(dev_dataset.post_list, desc="Generation"):
                    prompt = dev_data['prompt']
                    label = dev_data['label']
                    if "glm" in args.model_name_or_path:
                        prompt += tokenizer.mask_token
                        inputs = tokenizer(prompt, return_tensors="pt")
                        inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=args.max_length + args.max_length_generation)
                        inputs = inputs.to(device)
                        outputs = model.generate(**inputs,
                                                 max_new_tokens=args.max_length_generation,
                                                 eos_token_id=tokenizer.eop_token_id,
                                                 pad_token_id=tokenizer.pad_token_id,
                                                 do_sample=False,
                                                 num_return_sequences=args.num_return_sequences,
                                                 top_p=args.top_p,
                                                 temperature=args.temperature)
                    else:
                        inputs = tokenizer(prompt, add_special_tokens=False, return_token_type_ids=False, return_tensors="pt")
                        inputs = inputs.to(device)
                        outputs = model.generate(**inputs,
                                                 max_new_tokens=args.max_length_generation,
                                                 pad_token_id=tokenizer.pad_token_id,
                                                 do_sample=False,
                                                 num_return_sequences=args.num_return_sequences,
                                                 top_p=args.top_p,
                                                 temperature=args.temperature)
                        # outputs = text_generator(prompt, max_length=args.max_length_generation,
                        #                          do_sample=True, num_return_sequences=args.num_return_sequences,
                        #                          top_p=args.top_p, temperature=args.temperature)
                        # results = [output['generated_text'].split("答:", maxsplit=1)[1].replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "") for output in outputs]
                    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    results = [result.split("答:", maxsplit=1)[1] for result in results]

                    # metrics calculation
                    em_max = -1
                    f1_max = -1
                    for l in label:
                        for pred_text in results:
                            label_text = l['text']
                            em = 1 if pred_text == label_text else 0
                            f1 = calculate_f1(pred_text, label_text)
                            w.write(json.dumps({"prompt": prompt, "label": label_text,
                                                "pred": pred_text, "em": em, "f1": f1}, ensure_ascii=False)+"\n")
                            if em > em_max:
                                em_max = em
                            if f1 > f1_max:
                                f1_max = f1
                    ems.append(em_max)
                    f1s.append(f1_max)

        logger.info(f"em={np.mean(ems)}, f1={np.mean(f1s)}")
    else:
        sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler=sampler, batch_size=args.eval_batch_size)

        ppl_list = []
        input_ids_list = []
        label_list = []
        ls_list = []

        with torch.no_grad():
            for batch in tqdm(dev_dataloader, desc="Evaluation"):
                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].squeeze(1).to(device)
                labels = batch['labels'].squeeze(1).to(device)
                out = model(input_ids, attention_mask=attention_mask)
                ppls = preprocess_logits_for_metrics(out.logits, labels)
                input_ids_list.extend(batch['input_ids'].detach().cpu().tolist())
                ppl_list.extend(ppls.detach().cpu().tolist())
                label_list.extend(batch['label_str'])
                if args.task in ['chid', 'c3', 'iflytek', 'tnews']:
                    ls = np.array(batch['candidates']).transpose().tolist()
                    ls_list.extend(ls)
                else:
                    vals = list(dev_dataset.label_dict.values())
                    ls_list.extend([vals]*input_ids.shape[0])

        ct = 0
        ct_acc = 0
        ppls = []
        with open(output_filename, "w", encoding="utf-8") as w:
            for i, (input_ids, label, ls, ppl) in enumerate(zip(input_ids_list, label_list, ls_list, ppl_list)):
                ppls.append(ppl)
                prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
                if i % len(ls) == len(ls) - 1:
                    lidx = ls.index(label)
                    if np.argmin(ppls) == lidx:
                        ct_acc += 1
                    ct += 1
                    # cur_label = None
                    ppls = []
                w.write(json.dumps({"prompt": prompt, "pred": float(ppl), "label": label}, ensure_ascii=False) + "\n")

        logger.info(f"ppl={ct_acc/ct}")


def test():
    args = get_parser()
    logger.info(f"Parameters: {args}")

    set_seed(args.seed)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
    if "pangu" in args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_cache=False, trust_remote_code=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    prompts = [
        '今天晚上我在睡觉.........他想要做那些事..我就大大声骂他"不要吵我睡觉"!!!!!...他就跑出去了...还不接我电话',
        '首尔（：서울），正式名称为首尔特别市（서울 특별시），是韩国的首都，旧称汉城、汉阳等，「首尔」是韩语「서울」的汉语译音名称。2005年1月18日，该市市议会正式通过把其市的中文名称定为「首尔」，并把官方的所有出版物改用新名称，但部份华语地区还使用汉城。 首尔（서울），旧称南京（남경）、汉阳（한양）、汉城（한성）、京城（경성）。自从1945年二战结束韩国独立以后，当地民族情绪高涨，并将其首都改称为서울（英文为Seoul），韩语是「首都」或「京城」的意思，就如中国的首都称为“京”一样。因此，Seoul成为当时韩国国内唯一没有对应汉字名称的地名。中国大陆早期曾经根据韩语发音而音译作“苏乌”，但其后跟随其他华语地区，沿用这个城市在李氏朝鲜时的旧称“汉城”。然而，其他语言都翻译成“SEOUL”或类似的发音 这使日常沟通出现不少混乱，因为首尔市内有不少建筑物分别使用“首尔”或“汉城”为名称，但翻译成中文之后，这些地方名称会变得很相似。一个例子是：首尔市内同时有“汉城大学”及“首尔大学”，假若采用“汉城”作为 Seoul 的首都名称，会使两家大学的译名都变成了“汉城大学”。后来，中国人根据后者的英文名称，在其中文名称加上“国立”二字作分辨。但其实，首尔大学在韩国的知名度比汉城大学高很多，而大学本身的中文出版物，亦自称为“Seoul大学校”。但对於学校以外的机构，译名的问题就给他们造成困扰。2004年，韩国曾经有人发起运动要求解散首尔大学，韩国多份主要报章由於不清楚中国对两家“汉城大学”的区分方式，而多次在他们的中文版新闻中把首尔大学错译为“汉城大学”，而其企划部的柳根培亦无缘无故被“转校”成为汉城大学的发言人。 韩国政府从1992年起就一直在进行“서울”的中文名称的制订工作，拟订的对应中文包括“首尔”和“首午尔”。2005年1月18日，서울市议会通过使用与“SEOUL”发音相近的“首尔”作为这个城市的中文名称，并要求韩国政府公文、出版物、网站、机场和车站、公路路标的中文版本都使用“首尔”来代替“汉城”。但是迄今为止，在中文裏使用最多的还是具有500余年历史的“汉城”这个名称。香港及台湾的传媒大部份已经转用“首尔”这个新的名称；中国大陆官方正式宣布了改名这个消息，但并不常使用“首尔”这个新名。一些民办传媒及报刊先开始采用“首尔”，一些官方媒体（如新华网）也已经开始使用，还有一些处在过渡期，如中国中央电视台写成“汉城（首尔）”，不过中国大陆绝大部份出版物、媒体、政府及商业机关仍延续旧称。 有不少中国人质疑市议会是否有权更改本国首都的汉语名称，并指如果韩国首都的中文译名改变，将使华人世界对于韩国首都的称呼造成混乱。还有一个重要原因是“首尔”主要是根据现时汉语普通话的译音，但汉字是在汉语各方言间，以及韩、日、越南语的一些时候使用的，如果音译的话，会造成很多使用汉字的地区对“首尔”两字读音发生混乱，如粤语读为sau2 yi5，和韩语读音差别很大。而“首尔”两字按韩语中汉字的读音也成了「수이」（Su-i）。随著语音的发展，若干年后，即使普通话“首尔”两字的发音和韩语也可能对应不上，和众多西方音译词在各处的差别一样，可能会造成汉字使用者的困扰。有人提出如果根据韩语서울采用汉字“西尉”（韩语读作서울，即Seoul）则不会有此问题，可以在使用汉字的地区和时间上保持一致。 然而，韩国方面一直对中国大陆的这些想法非常不解。一来这城市是他们的城市，二来他们亦已多次透过各种沟道来解释这次改变的因由。 首尔是韩国首都汉城根据其英文“SEOUL”的音译所改的新的中文名称。 汉城市长李明博2005年1月19日举行记者招待会，宣布将首都的中文名称由“汉城”改为“首尔”。“汉城”一词不再使用。李明博市长的解说词是：绝大多数国家都将“SEOUL”按照与其英文标记相似的发音来称呼。如：汉语中的华盛顿、伦敦也都是根据这些地名的固有发音来标记的；只有汉城的中文名称一直沿用古代名称“汉城”。李明博市长向中国游说：“首尔”同汉城的韩语发音最为接近，并采用多用于外国地名的常用汉字，中国人也容易熟悉这一新名称。 很明显，李市长是从语音学角度切入问题的。但是如果就事论事的话，李明博等韩国官方人员的解释比较牵强。因为即使改换了“汉城”汉语的名称为“首尔”，也存在着翻译上及使用习惯上的混乱问题。况且，汉语中的外国地名也不都是以发音为根据翻译的，如英国的牛津、剑桥等并非完全是音译，美国的盐湖城(SsltLakeCity)、阿肯色州的小石城(LittleRock)等完全是意译。新西兰首都（Wellington）的官方中译名几乎就是错译——“威灵顿”显然比“惠灵顿”更贴切，但新西兰政府似乎从未提出任何异议。中文名称叫什么？应当尊重中国的传统使用习惯。 据考证，“汉城”的称呼沿用了韩国古代历史王朝的用法。1394年，李成桂将都城从开京迁移到了汉阳，正式命名为汉城(Hansung)。这一名称在汉语中至今已经使用了610年。二战结束以后，韩国将汉城称为韩国语的“首都”(Sieur)，英文音译为Seoul，但是韩国的书面汉字仍然写作汉城。因此，可以说，韩国这次改换首都的汉语名称绝非像表面上解释的那样简单，而是包含深刻的原因和其他方面复杂的考虑。 随着19世纪末民族主义的崛起，韩国国内就出现了不能正确认识本民族文化和客观评价中国文化对韩国民族文化影响，而摆脱汉语文化影响的思潮。韩国在二战以后，民族主义思潮进一步发展，曾以法律规定，以韩国的表音字为专用文字。从1970年起，韩国小学、中学教科书中的汉字被取消，完全使用表音文字。 随着韩国经济的崛起，这种极端的民族主义情绪进一步发展，在1988年汉城奥运会召开之前，韩国政府曾经下令取消所有牌匾上的汉字标记，以强调韩国的民族文化。 只是到了1999年2月，金大中总统才下令部分解除对汉字使用的限制。但对于这种解禁措施，韩国国内也存在着激烈的反对势力，他们担心这种措施将导致汉字的泛滥与韩国文字的消亡。 从某种意义上说，韩国改变“汉城”的中文名字是本国民族主义在新形势下的延续和发展的表现。 汉武帝时曾在朝鲜设立了四个郡。“汉城”是中国人六百年前至今一直习惯称谓的名字。韩国过去也一直用汉字写人名和地名。虽然“汉城”之名是由韩国古代的先人所起，但现代的韩国人总觉得不是本国的名称，总觉得“汉城”与中国有瓜葛，容易让人联想到中国的汉朝。对于汉朝，一些韩国人又缺乏正确的历史观，认为是对朝鲜的侵略。正式在这种不良心理情结的指导下，韩国才有意将“汉城”的中文译名更改为“首尔”。 韩国官方为这次改名的解释，仅仅是表面的，这是以国际惯例和便于国际交往来掩盖更加深层的阴暗心理情结，努力摆脱汉字文化对韩国深厚影响的一种尝试。 叫了610年的汉城一下子改名，叫着真让人感觉别扭！就好比纽约（New Youk）突然让叫“牛月克”。你能习惯吗？我是习惯不了。我在很多方面敬佩和尊重韩国人，但是在这一点上，我B4韩国人。太小家子气了！ 不可否认，朝鲜民族在历史上深受日本侵略者的奴役之苦，大力弘扬朝鲜本民族的文化，加强自身民族文化的优势地位，努力摆脱外来文化影响等措施，有可以理解的一面，不应该随意扣上狭隘的帽子。 但是，另一方面，韩国自身在保持和发扬本民族文化的同时，对外来文化，特别是博大精深的中国文化，也应该采取扬弃的态度，不应该不分好坏一概拒绝。其实，博大精深的中华文化对朝鲜民族文化的产生、发展都起到了巨大的贡献作用。 在具体对待这次改名的问题上，韩国有权利更改本国首都的汉语译名，但是中国以及其他汉语权的国家也有权接受或不接受这个新译名。接受与不接受，这也是中国与其他汉语国家应有的权利，韩国应当予以尊重。因为对中国等这些国家来说，不仅仅是一个译法上的问题，而涉及了历史习惯、经济费用等多方面的问题。'
    ]
    # prompt = '今天晚上我在睡觉.........他想要做那些事..我就大大声骂他"不要吵我睡觉"!!!!!...他就跑出去了...还不接我电话'
    # target = '首尔（：서울），正式名称为首尔特别市（서울 특별시），是韩国的首都，旧称汉城、汉阳等，「首尔」是韩语「서울」的汉语译音名称。2005年1月18日，该市市议会正式通过把其市的中文名称定为「首尔」，并把官方的所有出版物改用新名称，但部份华语地区还使用汉城。 首尔（서울），旧称南京（남경）、汉阳（한양）、汉城（한성）、京城（경성）。自从1945年二战结束韩国独立以后，当地民族情绪高涨，并将其首都改称为서울（英文为Seoul），韩语是「首都」或「京城」的意思，就如中国的首都称为“京”一样。因此，Seoul成为当时韩国国内唯一没有对应汉字名称的地名。中国大陆早期曾经根据韩语发音而音译作“苏乌”，但其后跟随其他华语地区，沿用这个城市在李氏朝鲜时的旧称“汉城”。然而，其他语言都翻译成“SEOUL”或类似的发音 这使日常沟通出现不少混乱，因为首尔市内有不少建筑物分别使用“首尔”或“汉城”为名称，但翻译成中文之后，这些地方名称会变得很相似。一个例子是：首尔市内同时有“汉城大学”及“首尔大学”，假若采用“汉城”作为 Seoul 的首都名称，会使两家大学的译名都变成了“汉城大学”。后来，中国人根据后者的英文名称，在其中文名称加上“国立”二字作分辨。但其实，首尔大学在韩国的知名度比汉城大学高很多，而大学本身的中文出版物，亦自称为“Seoul大学校”。但对於学校以外的机构，译名的问题就给他们造成困扰。2004年，韩国曾经有人发起运动要求解散首尔大学，韩国多份主要报章由於不清楚中国对两家“汉城大学”的区分方式，而多次在他们的中文版新闻中把首尔大学错译为“汉城大学”，而其企划部的柳根培亦无缘无故被“转校”成为汉城大学的发言人。 韩国政府从1992年起就一直在进行“서울”的中文名称的制订工作，拟订的对应中文包括“首尔”和“首午尔”。2005年1月18日，서울市议会通过使用与“SEOUL”发音相近的“首尔”作为这个城市的中文名称，并要求韩国政府公文、出版物、网站、机场和车站、公路路标的中文版本都使用“首尔”来代替“汉城”。但是迄今为止，在中文裏使用最多的还是具有500余年历史的“汉城”这个名称。香港及台湾的传媒大部份已经转用“首尔”这个新的名称；中国大陆官方正式宣布了改名这个消息，但并不常使用“首尔”这个新名。一些民办传媒及报刊先开始采用“首尔”，一些官方媒体（如新华网）也已经开始使用，还有一些处在过渡期，如中国中央电视台写成“汉城（首尔）”，不过中国大陆绝大部份出版物、媒体、政府及商业机关仍延续旧称。 有不少中国人质疑市议会是否有权更改本国首都的汉语名称，并指如果韩国首都的中文译名改变，将使华人世界对于韩国首都的称呼造成混乱。还有一个重要原因是“首尔”主要是根据现时汉语普通话的译音，但汉字是在汉语各方言间，以及韩、日、越南语的一些时候使用的，如果音译的话，会造成很多使用汉字的地区对“首尔”两字读音发生混乱，如粤语读为sau2 yi5，和韩语读音差别很大。而“首尔”两字按韩语中汉字的读音也成了「수이」（Su-i）。随著语音的发展，若干年后，即使普通话“首尔”两字的发音和韩语也可能对应不上，和众多西方音译词在各处的差别一样，可能会造成汉字使用者的困扰。有人提出如果根据韩语서울采用汉字“西尉”（韩语读作서울，即Seoul）则不会有此问题，可以在使用汉字的地区和时间上保持一致。 然而，韩国方面一直对中国大陆的这些想法非常不解。一来这城市是他们的城市，二来他们亦已多次透过各种沟道来解释这次改变的因由。 首尔是韩国首都汉城根据其英文“SEOUL”的音译所改的新的中文名称。 汉城市长李明博2005年1月19日举行记者招待会，宣布将首都的中文名称由“汉城”改为“首尔”。“汉城”一词不再使用。李明博市长的解说词是：绝大多数国家都将“SEOUL”按照与其英文标记相似的发音来称呼。如：汉语中的华盛顿、伦敦也都是根据这些地名的固有发音来标记的；只有汉城的中文名称一直沿用古代名称“汉城”。李明博市长向中国游说：“首尔”同汉城的韩语发音最为接近，并采用多用于外国地名的常用汉字，中国人也容易熟悉这一新名称。 很明显，李市长是从语音学角度切入问题的。但是如果就事论事的话，李明博等韩国官方人员的解释比较牵强。因为即使改换了“汉城”汉语的名称为“首尔”，也存在着翻译上及使用习惯上的混乱问题。况且，汉语中的外国地名也不都是以发音为根据翻译的，如英国的牛津、剑桥等并非完全是音译，美国的盐湖城(SsltLakeCity)、阿肯色州的小石城(LittleRock)等完全是意译。新西兰首都（Wellington）的官方中译名几乎就是错译——“威灵顿”显然比“惠灵顿”更贴切，但新西兰政府似乎从未提出任何异议。中文名称叫什么？应当尊重中国的传统使用习惯。 据考证，“汉城”的称呼沿用了韩国古代历史王朝的用法。1394年，李成桂将都城从开京迁移到了汉阳，正式命名为汉城(Hansung)。这一名称在汉语中至今已经使用了610年。二战结束以后，韩国将汉城称为韩国语的“首都”(Sieur)，英文音译为Seoul，但是韩国的书面汉字仍然写作汉城。因此，可以说，韩国这次改换首都的汉语名称绝非像表面上解释的那样简单，而是包含深刻的原因和其他方面复杂的考虑。 随着19世纪末民族主义的崛起，韩国国内就出现了不能正确认识本民族文化和客观评价中国文化对韩国民族文化影响，而摆脱汉语文化影响的思潮。韩国在二战以后，民族主义思潮进一步发展，曾以法律规定，以韩国的表音字为专用文字。从1970年起，韩国小学、中学教科书中的汉字被取消，完全使用表音文字。 随着韩国经济的崛起，这种极端的民族主义情绪进一步发展，在1988年汉城奥运会召开之前，韩国政府曾经下令取消所有牌匾上的汉字标记，以强调韩国的民族文化。 只是到了1999年2月，金大中总统才下令部分解除对汉字使用的限制。但对于这种解禁措施，韩国国内也存在着激烈的反对势力，他们担心这种措施将导致汉字的泛滥与韩国文字的消亡。 从某种意义上说，韩国改变“汉城”的中文名字是本国民族主义在新形势下的延续和发展的表现。 汉武帝时曾在朝鲜设立了四个郡。“汉城”是中国人六百年前至今一直习惯称谓的名字。韩国过去也一直用汉字写人名和地名。虽然“汉城”之名是由韩国古代的先人所起，但现代的韩国人总觉得不是本国的名称，总觉得“汉城”与中国有瓜葛，容易让人联想到中国的汉朝。对于汉朝，一些韩国人又缺乏正确的历史观，认为是对朝鲜的侵略。正式在这种不良心理情结的指导下，韩国才有意将“汉城”的中文译名更改为“首尔”。 韩国官方为这次改名的解释，仅仅是表面的，这是以国际惯例和便于国际交往来掩盖更加深层的阴暗心理情结，努力摆脱汉字文化对韩国深厚影响的一种尝试。 叫了610年的汉城一下子改名，叫着真让人感觉别扭！就好比纽约（New Youk）突然让叫“牛月克”。你能习惯吗？我是习惯不了。我在很多方面敬佩和尊重韩国人，但是在这一点上，我B4韩国人。太小家子气了！ 不可否认，朝鲜民族在历史上深受日本侵略者的奴役之苦，大力弘扬朝鲜本民族的文化，加强自身民族文化的优势地位，努力摆脱外来文化影响等措施，有可以理解的一面，不应该随意扣上狭隘的帽子。 但是，另一方面，韩国自身在保持和发扬本民族文化的同时，对外来文化，特别是博大精深的中国文化，也应该采取扬弃的态度，不应该不分好坏一概拒绝。其实，博大精深的中华文化对朝鲜民族文化的产生、发展都起到了巨大的贡献作用。 在具体对待这次改名的问题上，韩国有权利更改本国首都的汉语译名，但是中国以及其他汉语权的国家也有权接受或不接受这个新译名。接受与不接受，这也是中国与其他汉语国家应有的权利，韩国应当予以尊重。因为对中国等这些国家来说，不仅仅是一个译法上的问题，而涉及了历史习惯、经济费用等多方面的问题。'
    prefix = "回答："
    # prompt = '上联：东风执笔点龙睛，看幸福指数，天天向上'
    # target = '春雨唤梅开岁首，欣浪漫滨城，步步登高'
    # prefix = "下联："

    # # train
    # if "glm" in args.model_name_or_path:
    #     encoded_prompt = tokenizer(prompt, prefix + tokenizer.mask_token)
    #     prompt_length = len(encoded_prompt['input_ids'])
    #     label_length = len(tokenizer.tokenize(target)) + 1
    #     if prompt_length + label_length > args.max_length:
    #         num_tokens_to_remove = prompt_length + label_length - args.max_length
    #         for _ in range(num_tokens_to_remove):
    #             if prompt_length > label_length:
    #                 prompt_length -= 1
    #             else:
    #                 label_length -= 1
    #     else:
    #         label_length = args.max_length - prompt_length
    #     assert prompt_length > 0
    #     assert label_length > 0
    #     assert prompt_length + label_length == args.max_length
    #     inputs = tokenizer(prompt, prefix + tokenizer.mask_token,
    #                        max_length=prompt_length, truncation="only_first",
    #                        return_tensors="pt", return_token_type_ids=False)
    #     inputs_glm = tokenizer.build_inputs_for_generation(inputs, targets=target,
    #                                                        max_gen_length=label_length, padding=True)
    #     outputs = model(**inputs_glm)
    # else:
    #     pass

    # # generation (model.generate)
    # if "glm" in args.model_name_or_path:
    #     encoded_prompt = tokenizer(prompt, prefix + tokenizer.mask_token)
    #     prompt_length = len(encoded_prompt['input_ids'])
    #     inputs = tokenizer(prompt, prefix + tokenizer.mask_token,
    #                        max_length=min(prompt_length, args.max_length),
    #                        truncation="only_first",
    #                        return_tensors="pt",
    #                        return_token_type_ids=False)
    #     max_gen_length = args.max_length - inputs['input_ids'].shape[1]
    #     inputs_glm = tokenizer.build_inputs_for_generation(inputs,
    #                                                        max_gen_length=max_gen_length, padding=True)
    #     outputs = model.generate(**inputs_glm,
    #                              max_new_tokens=args.max_length_generation,
    #                              eos_token_id=tokenizer.eop_token_id,
    #                              pad_token_id=tokenizer.pad_token_id,
    #                              do_sample=False,
    #                              num_return_sequences=args.num_return_sequences,
    #                              top_p=args.top_p,
    #                              temperature=args.temperature)
    # else:
    #     inputs = tokenizer(prompt, tokenizer.sep_token + prefix, max_length=args.max_length,
    #                        truncation="longest_first", add_special_tokens=False,
    #                        return_tensors="pt", return_token_type_ids=False)
    #     outputs = model.generate(**inputs,
    #                              max_new_tokens=args.max_length_generation,
    #                              pad_token_id=tokenizer.pad_token_id,
    #                              do_sample=False,
    #                              num_return_sequences=args.num_return_sequences,
    #                              top_p=args.top_p,
    #                              temperature=args.temperature)
    # results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # results = [result.split(prefix, maxsplit=1)[1].strip() for result in results]

    # generation (transformers GenerationPipeline)
    text_generator = TextGenerationPipeline(model, tokenizer)
    outputs = text_generator([prompt + tokenizer.sep_token + prefix for prompt in prompts],
                             max_new_tokens=args.max_length_generation,
                             bos_token_id=tokenizer.bos_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.pad_token_id,
                             do_sample=True, num_return_sequences=args.num_return_sequences,
                             top_p=args.top_p, temperature=args.temperature)
    results = [output[0]['generated_text'].split(prefix, maxsplit=1)[1] for output in outputs]


if __name__ == "__main__":
    # main()
    test()
