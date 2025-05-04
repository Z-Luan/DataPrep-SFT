from langdetect import detect
from langdetect import detect_langs
import langid
import json
import evaluate
from tqdm import tqdm
import numpy as np
import time
import re
import jieba
from GPT4 import Llm
from collections import Counter
from collections import Counter
import concurrent.futures

def load_from_jsonl(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

def save_to_jsonl(dataset, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for data in dataset:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

# 基于启发式规则的质量过滤
#---- 超链接过滤 ----#
def hyperlink_filtering(dataset, if_conduct = True):
    if not if_conduct:
        return dataset
    
    url_pattern = r"https?://\S+|www\.\S+"
    clean_pattern = r"[^。！？]*?(https?://\S+|www\.\S+)[^。！？]*?[。！？]?"

    dataset_f = []
    for data in tqdm(dataset, desc="HyperLink Filtering"):
        title = data['title']
        answers = data['answers']

        answers_f = []
        for answer in answers:
            if re.search(url_pattern, answer):
                answer_clean_hl = re.sub(clean_pattern, "", answer)
                if len(answer_clean_hl) != 0:
                    answers_f.append(answer_clean_hl)
            else:
                answers_f.append(answer)

        dataset_f.append({
            'title': title,
            'answers': answers_f
        })
    
    return dataset_f

#---- 语言过滤 ----#
def language_based_filtering(dataset, if_conduct = True): 
    if not if_conduct:
        return dataset
    
    dataset_f = []
    for data in tqdm(dataset, desc="Language Filtering"):
        title = data['title']
        answers = data['answers']

        title_language = detect(title)
        if title_language not in ["zh-cn", "ko"]: # 中文语言可能会被错误检测为韩语
            print(f"问题“{title}”是{title_language}语言，不是中文语言！")
            continue

        answers_f = []
        for answer in answers: 
            try:
                answer_language = detect(answer) # 仅数字符号会报错
            except:
                answers_f.append(answer)
                continue
            if answer_language in ["zh-cn", "ko"]:
                answers_f.append(answer)
            else:
                print(f"问题“{title}”的回答“{answer}”是{answer_language}语言，不是中文语言！")
        
        dataset_f.append({
            'title': title,
            'answers': answers_f
        })
    
    return dataset_f

#---- 断句过滤 ----#
def stop_symbol_based_filtering(dataset, if_conduct = True): 
    if not if_conduct:
        return dataset
    
    stop_symbols = ['。', '？', '！', '.', '?', '!', '~']

    dataset_f = []
    for data in tqdm(dataset, desc="StopSymbol Filtering"):
        title = data['title']
        answers = data['answers']

        answers_f = []
        for answer in answers:
            if answer[-1] in stop_symbols:
                answers_f.append(answer)
            elif len(list(jieba.cut(answer))) >= 3:
                answers_f.append(answer + "。")
            else: # 缺少结尾停止符且长度过短的回答很可能不完整
                print(f"问题“{title}”的回答“{answer}”不完整！")

        dataset_f.append({
            'title': title,
            'answers': answers_f
        })
    
    return dataset_f
    
#---- 困惑度过滤 ----#
# 大模型对领域数据集较为“陌生”，困惑度整体偏高
def ppl_based_filtering(dataset, if_conduct = True): 
    if not if_conduct:
        return dataset
    
    ppl_metric = evaluate.load("perplexity", module_type="metric")
    
    dataset_f = []
    for data in tqdm(dataset, desc="Perplexity Filtering"):
        title = data['title']
        answers = data['answers']

        ppl = ppl_metric.compute(predictions=[title] + answers, model_id="Qwen/Qwen2.5-3B", device='cuda')
        ppl_title = ppl['perplexities'][0]
        ppl_answers = ppl['perplexities'][1:]
        sigma_a = np.std(ppl_answers)
        
        answers_f = []
        for idx, answer in enumerate(answers):
            ppl_answer = ppl_answers[idx]
            
            if ppl_answer - ppl_title <= 2 * sigma_a: # 过滤不相关答案
                answers_f.append(answer)
            else:
                print(f"问题“{title}”的回答“{answer}”低质量！")
        
        dataset_f.append({
            'title': title,
            'answers': answers_f
        })
    
    return dataset_f

# 基于分类器的质量过滤
#---- GPT4 ----# 
def llm_filtering(dataset, sampling_frequency=3, if_conduct=True):
    if not if_conduct:
        return dataset

    llm = Llm()
    system_prompt = "你是一个专业的回答质量评估助手。给定一个问题和多个回答，你的任务是遵循**1.准确性：回答应与问题紧密相关，信息无误且权威；2.全面性：回答应包含足够的信息，避免过于简略或片面；3.简洁性：在保证信息完整的情况下，尽量简洁明了**三条标准，从多个回答中选择最佳回答。请直接返回选择的最佳回答，不要输出解释或额外信息。"
    query_prompt = "问题：{}\n回答列表：{}"

    # ----------------------- #
    def process_one(query):
        answer_f, logprobs = llm.generate(query, system_prompt) # GPT4
        return answer_f
    # ----------------------- #

    dataset_f = []
    for data in tqdm(dataset, desc="LLM Filtering"):
        title = data['title']
        answers = data['answers']

        answer_f_candidate = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_one, query_prompt.format(title, answers)) for _ in range(sampling_frequency)]
            for future in concurrent.futures.as_completed(futures):
                answer_f_candidate.append(future.result().strip("'"))
        
        counter = Counter(answer_f_candidate)
        dataset_f.append({
            'title': title,
            'answers': counter.most_common(1)[0][0]
        })

    return dataset_f

# def llm_filtering(dataset, sampling_frequency=3, if_conduct = True):
    if not if_conduct:
        return dataset
    
    llm = Llm()
    system_prompt = "你是一个专业的回答质量评估助手。给定一个问题和多个回答，你的任务是遵循**1.准确性：回答应与问题紧密相关，信息无误且权威；2.全面性：回答应包含足够的信息，避免过于简略或片面；3.简洁性：在保证信息完整的情况下，尽量简洁明了**三条标准，从多个回答中选择最佳回答。请直接返回选择的最佳回答，不要输出解释或额外信息。"
    query_prompt = "问题：{}\n回答列表：{}"

    dataset_f = []
    for data in tqdm(dataset, desc="LLM Filtering"):
        title = data['title']
        answers = data['answers']

        answer_f_candidate = []
        for _ in range(sampling_frequency): # Self-Consistency
            answer_f, logprobs = llm.generate(query_prompt.format(title, answers), system_prompt) # GPT4
            answer_f_candidate.append(answer_f.strip("'"))

        counter = Counter(answer_f_candidate)
        dataset_f.append({
            'title': title,
            'answers': counter.most_common(1)[0][0]
        })
    
    return dataset_f


if __name__ == "__main__": 
    dataset_path = "./dataset/数据收集.jsonl"
    dataset = load_from_jsonl(dataset_path)

    dataset_f_1 = hyperlink_filtering(dataset)
    dataset_f_2 = language_based_filtering(dataset_f_1)
    dataset_f_3 = stop_symbol_based_filtering(dataset_f_2)
    dataset_f_4 = ppl_based_filtering(dataset_f_3, if_conduct = False)
    dataset_f_5 = llm_filtering(dataset_f_4)
