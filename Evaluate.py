import json
from tqdm import tqdm
from rouge_score import rouge_scorer
import bert_score
import numpy as np
import argparse
from rouge_chinese import Rouge
import jieba

def read_jsonl(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

def compute_rouge_chinese(predictions, references): # 支持中文分句、优化内存占用
    rouge = Rouge()
    scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}

    for pred, ref in zip(predictions, references):
        score = rouge.get_scores(pred, ref)[0] 
        for key in scores:
            scores[key].append(score[key]['f'])  # 考虑F1分数

    return {
        'ROUGE-1': np.mean(scores['rouge-1']),
        'ROUGE-2': np.mean(scores['rouge-2']),
        'ROUGE-L': np.mean(scores['rouge-l']),
    }

def compute_bertscore(predictions, references):
    P, R, F1 = bert_score.score(predictions, references, lang="zh", verbose=True)
    return {
        'BERTScore-P': P.mean().item(),
        'BERTScore-R': R.mean().item(),
        'BERTScore-F1': F1.mean().item(),
    }


if __name__ == "__main__":
    path = "/home/SFTandRLHF/dataset/test_w_response_Qwen-2.5B.jsonl"

    dataset = read_jsonl(path)

    predictions = []
    references = []
    for data in tqdm(dataset, desc="Evaluating"):
        pre = data['response'].strip()
        ref = data['output'].strip()

        predictions.append(pre)
        references.append(ref)

    rouge_results = compute_rouge_chinese(predictions, references)
    bert_results = compute_bertscore(predictions, references)

    print("\n===== Evaluation Results =====")
    for k, v in {**rouge_results, **bert_results}.items(): # **用于字典解包
        print(f"{k}: {v:.4f}")

    #-----------Base-----------#
    # ===== Evaluation Results =====
    # ROUGE-1: 0.0014
    # ROUGE-2: 0.0001
    # ROUGE-L: 0.0014
    # BERTScore-P: 0.6268
    # BERTScore-R: 0.6736
    # BERTScore-F1: 0.6471

    #-----------SFT-----------#
    # ===== Evaluation Results =====
    # ROUGE-1: 0.0068
    # ROUGE-2: 0.0023
    # ROUGE-L: 0.0066
    # BERTScore-P: 0.6854
    # BERTScore-R: 0.6583
    # BERTScore-F1: 0.6681
