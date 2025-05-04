import json
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

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

def create_minhash(data):
    minhash = MinHash(num_perm=128)  # num_perm哈希函数数量
    for d in data:
        minhash.update(d.encode('utf8'))
    return minhash

def duplication(dataset):
    lsh = MinHashLSH(threshold=0.9, num_perm=128)  # threshold相似度阈值

    dataset_d = []
    for idx, data in enumerate(dataset):
        title = data['title']

        minhash = create_minhash(list(title))
        results = lsh.query(minhash)
        if  len(results) == 0:  
            dataset_d.append(dataset[idx])
            lsh.insert(len(dataset_d) - 1, minhash)
        else:
            print(f"当前问题: \n{title}\n相似问题: ")
            for result in results:
                print(dataset_d[result]['title'])
            print("\n")

    return dataset_d


if __name__ == "__main__": 
    dataset_path = "./dataset/数据收集.jsonl"
    dataset = load_from_jsonl(dataset_path)

    dataset_d = duplication(dataset)
    print(f"去重前数据量: {len(dataset)}, 去重后数据量: {len(dataset_d)}")
