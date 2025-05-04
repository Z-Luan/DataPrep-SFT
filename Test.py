from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights,infer_auto_device_map
import torch
import numpy as np
import os
import json
import re
import argparse
import random
from tqdm import tqdm
import string

def setup(model_name):
    cuda_list = args.device.split(',') 
    memory = '45GiB' 
    if model_name == "Qwen-2.5B":
        model_path = "/data/Model/Qwen2.5-3B"
    elif model_name == "Qwen-2.5B-SFT":
        model_path = "/home/SFTandRLHF/OpenRLHF/checkpoint/qwen2.5-3b-sft"

    max_memory = {int(cuda):memory for cuda in cuda_list}
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype = torch.float16)
    device_map = infer_auto_device_map(model, max_memory = max_memory)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map = device_map, torch_dtype = torch.float16, use_safetensors = True) 

    return tokenizer, model

def read_json(file_path):
    with open(file_path, encoding = 'utf-8') as f:
        content = f.read()
    dataset = json.loads(content)

    return dataset

def save_jsonl(dataset, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for data in dataset:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen-2.5B-SFT")
    parser.add_argument('--device', type=str, default="7")
    args = parser.parse_args()

    tokenizer, model = setup(args.model_name)
    dataset = read_json("/home/SFTandRLHF/dataset/test.json")
    
    dataset_w_response = []
    for data in tqdm(dataset):
        data_w_response = data.copy()
        question = data['input']
        ground_truth = data['output']

        chat = [{"role": "user", "content": question}]
        query = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        model.eval()
        input_participle = tokenizer(query, return_tensors="pt")
        input_ids = input_participle['input_ids'].to(model.device)

        output_participle = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_new_tokens=128) # do_sample=False 贪婪解码 max_new_tokens=128
        output = tokenizer.batch_decode(output_participle, skip_special_tokens = True)[0]

        response = output.rpartition("assistant")[2].strip()
        data_w_response['response'] = response

        dataset_w_response.append(data_w_response)

    save_jsonl(dataset_w_response, f"/home/SFTandRLHF/dataset/test_w_response_{args.model_name}.jsonl")

    