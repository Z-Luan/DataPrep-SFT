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
## 文件读取 ##
from Quality_Filtering import load_from_jsonl, save_to_jsonl
## 质量过滤 ##
from Quality_Filtering import hyperlink_filtering, language_based_filtering, stop_symbol_based_filtering, ppl_based_filtering, llm_filtering
## 去重 ##
from De_Duplication import duplication
## 隐私保护 ##
from Privacy_Reduction import privacy_reduction


dataset_path = "./dataset/数据收集.jsonl"
dataset = load_from_jsonl(dataset_path)
print(f"原始数据量: {len(dataset)}")

# 质量过滤
dataset_qf_1 = hyperlink_filtering(dataset)
dataset_qf_2 = language_based_filtering(dataset_qf_1)
dataset_qf_3 = stop_symbol_based_filtering(dataset_qf_2)
dataset_qf_4 = ppl_based_filtering(dataset_qf_3, if_conduct = False)
dataset_qf_5 = llm_filtering(dataset_qf_4)
save_to_jsonl(dataset_qf_5, "./dataset/数据收集_QF.jsonl")
print(f"质量过滤后数据量: {len(dataset_qf_5)}")

# 去重
dataset_d = duplication(dataset_qf_5)
save_to_jsonl(dataset_d, "./dataset/数据收集_QF_D.jsonl")
print(f"去重后数据量: {len(dataset_d)}")

# 隐私保护
dataset_pr = privacy_reduction(dataset_d)
save_to_jsonl(dataset_pr, "./dataset/数据收集_QF_D_PR.jsonl")
print(f"隐私保护后数据量: {len(dataset_pr)}")
