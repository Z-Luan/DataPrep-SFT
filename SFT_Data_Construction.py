from datasets import Dataset, DatasetDict
## 文件读取 ##
from Quality_Filtering import load_from_jsonl, save_to_jsonl
import json
import time
import random
random.seed(42)


dataset_path = "./dataset/数据收集_QF_D_PR.jsonl"
dataset = load_from_jsonl(dataset_path)
print(f"原始数据量: {len(dataset)}")
random.shuffle(dataset)

# 按8:2的比例划分训练集和测试集
split_idx = int(0.8 * len(dataset))
train_items = dataset[:split_idx]
test_items = dataset[split_idx:]

train_dataset = []
for train_item in train_items:
    q1, q2 = train_item["title"].split("\U0001F004", 1)
    query = f"问题：{q1.strip()} 具体情况如下：{q2.strip()}"

    train_dataset.append({
        "input": query,
        "output": train_item["answer"]
    })

test_dataset = []
for test_item in test_items:
    q1, q2 = test_item["title"].split("\U0001F004", 1)
    query = f"问题：{q1.strip()} 具体情况如下：{q2.strip()}"

    test_dataset.append({
        "input": query,
        "output": test_item["answer"]
    })

with open("./dataset/train.json", "w", encoding="utf-8") as f:
    json.dump(train_dataset, f, ensure_ascii=False, indent=2)

with open("./dataset/test.json", "w", encoding="utf-8") as f:
    json.dump(test_dataset, f, ensure_ascii=False, indent=2)

print(f"训练集数量：{len(train_dataset)}")
print(f"测试集数量：{len(test_dataset)}")

train_dataset = Dataset.from_list(train_dataset)
test_dataset = Dataset.from_list(test_dataset)

dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# 保存到本地
save_path = "./dataset/SFT_dataset"
dataset_dict.save_to_disk(save_path)
