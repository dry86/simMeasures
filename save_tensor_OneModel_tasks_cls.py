import os
import time
import json5
import torch
import pandas as pd
from transformers import AutoTokenizer
from utils import extract_prompts
from getHiddenStates import load_model, tokens_get_hidden_states
import numpy as np

def get_padding_length_and_prompt(task, tokenizer1, lang, device):

    # 加载tokenizer
    # tokenizer1 = AutoTokenizer.from_pretrained(model1_path)

    # 使用 eos_token 作为 pad_token
    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer1.padding_side = "right"   # 使用EOS时,向右填充

    token1 = []
    # 读取数据文件
    prompts = extract_prompts(task, lang)
    for task_id, prompt in prompts:
        # print(f"Task ID: {task_id}")

        inputs_model1 = tokenizer1(prompt, return_tensors='pt').to(device)
        token1.append(inputs_model1['input_ids'].cpu().numpy())

    lengths1 = [len(seq[0]) for seq in token1]

    # 使用numpy的percentile函数直接计算90%分位数
    percentile_90 = round(np.percentile(lengths1, 90))
    print(f"90% percentile for model: {percentile_90}")
    return percentile_90, prompts

def main(task, model, tokenizer1, model_idx, padding_len, prompts, lang, device1, batch_size=1):

    # 加载模型和tokenizer
    # model, tokenizer1 = load_model(model1_path, device1)x

    # 使用 eos_token 作为 pad_token
    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer1.padding_side = "right"   # 使用EOS时,向右填充

    save_dir = model_path + "/pt_file" + f"/{task}/" +  lang + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 存储每次获取的 last_token_hidden_states
    accumulated_hidden_states = []
    batch_counter = 0
    batch_idx = 1

    print(f"\tbatch process start!")
    for task_id, prompt in prompts:
        # print(f"Task ID: {task_id}")

        inputs = tokenizer1(prompt, 
                            return_tensors='pt',
                            padding='max_length', 
                            max_length=padding_len, 
                            truncation=True
                            ).to(device1)

        # 获取隐藏层输出
        hidden_states = tokens_get_hidden_states(model, inputs)

        # 获取每一层最后一个有效(非padding)token
        last_non_padding_index = inputs['attention_mask'].sum(dim=1) - 1
        last_token_hidden_states = [layer_output[torch.arange(batch_size), last_non_padding_index, :].squeeze(0).cpu() for layer_output in hidden_states]

        # 将每层的 last_token_hidden_states 添加到累积列表
        accumulated_hidden_states.append(last_token_hidden_states)
        batch_counter += batch_size

        # 当累积数量达到1000时，保存hidden states并清空累积
        if batch_counter >= 1000:
            # 将累积的 hidden states 转换为张量 (1000, num_layers, hidden_size)
            concatenated_hidden_states = torch.stack([torch.stack(states) for states in accumulated_hidden_states])

            # 进行形状变换，使得保留33层, 每层形状为(1000, 4096)
            concatenated_hidden_states = concatenated_hidden_states.permute(1, 0, 2)  # 变换为 (num_layers, 1000, 4096)

            # 保存拼接的 hidden states 到文件
            torch.save(concatenated_hidden_states, f"{save_dir}{model_idx}_batch_{batch_idx}.pt")
            print(f"\tbatch_{batch_idx} saved!")
            batch_idx = batch_idx + 1

            # 清空累积列表和计数器
            accumulated_hidden_states.clear()
            batch_counter = 0

            break # only first 1k
        
        del hidden_states, last_token_hidden_states
        torch.cuda.empty_cache()
            
    # 如果循环结束后仍有未保存的hidden states
    if accumulated_hidden_states:
        concatenated_hidden_states = torch.stack([torch.stack(states) for states in accumulated_hidden_states])
        concatenated_hidden_states = concatenated_hidden_states.permute(1, 0, 2)  # 变换为 (33, 1000, 4096)
        torch.save(concatenated_hidden_states, f"{save_dir}{model_idx}_batch_{batch_idx}.pt")

if __name__ == "__main__":
    """
    how to use:
    python save_tensor_OneModel_tasks.py
    output:
    1. 每个任务的tensor文件, 保存在model_path/pt_file/task/lang/ 下
    2. 每个任务的tensor文件, 命名为 model_idx_batch_1.pt, model_idx_batch_2.pt, ...
    注意⛔️:
        是否 only save first 1k
    """
    # 记录开始时间
    start_time = time.time()  

    device_model = torch.device("cuda:0")

    # 参数设置
    configs = json5.load(open('/root/projects/simMeasures/config/config-save-tasks.json5'))

    for config in configs:
        task = config.get('task')
        model_path = config.get('model_path')
        model_idx = os.path.basename(model_path)
        langs = config.get('lang')

        print(f"task list: {task}, {model_idx}, {langs}")
    print("-"*100)

    for config in configs:
        tasks = config.get('task')
        model_path = config.get('model_path')
        model_idx = os.path.basename(model_path)

        langs = config.get('lang')
        # 确保langs是列表形式
        if isinstance(langs, str):
            langs = [langs]

        model, tokenizer1 = load_model(model_path, device_model)
        for task in tasks:
            for lang in langs:
                print(f"Current work: {model_idx}, {task}, {lang}")
                
                padding_length, prompts = get_padding_length_and_prompt(task, tokenizer1, lang, device_model)
                main(task, model, tokenizer1, model_idx, padding_length, prompts, lang, device_model)
        
        del model, tokenizer1
        torch.cuda.empty_cache()

        # 记录结束时间
        end_time = time.time()
        # 计算并打印程序运行时间
        elapsed_time = (end_time - start_time) / 60
        print(f"Program runtime: {elapsed_time:.2f} mins")  