import json
from tqdm import tqdm
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine

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

def privacy_reduction(dataset):
    nlp_engine = SpacyNlpEngine(models=[{"lang_code": "en", "model_name": "en_core_web_sm"}, {"lang_code": "zh", "model_name": "zh_core_web_sm"}])
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    anonymizer = AnonymizerEngine()

    dataset_pr = []
    for data in tqdm(dataset, desc="Privacy Reduction"):
        title = data['title']
        answer = data['answer']

        answer_pr = []
        results = analyzer.analyze(text=answer, entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "NRP"], score_threshold=0.9, language="zh")
        anonymized_answer = anonymizer.anonymize(text=answer, analyzer_results=results)
        anonymized_answer_text = anonymized_answer.text
        anonymized_answer_items = anonymized_answer.items

        if len(anonymized_answer_items) != 0:
            print(f"原回答：{answer}")
            print(f"脱敏回答：{anonymized_answer_text}")

        answer_pr = anonymized_answer_text

        dataset_pr.append({
            'title': title,
            'answer': answer_pr
        })
        
    return dataset_pr


if __name__ == "__main__": 
    dataset_path = "./dataset/数据收集.jsonl"
    dataset = load_from_jsonl(dataset_path)

    dataset_pr = privacy_reduction(dataset)
    