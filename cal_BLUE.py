import torch
from getHiddenStates import load_model, get_hidden_states
import numpy as np
import jsonlines
from example.Topology import *
import evaluate


def calculate_PBM_BLUE(pred1, pred2, references):
    # 计算模型1的BLEU分数
    results_model1 = bleu.compute(predictions=pred1, references=references)
    print(f"\t{'模型1的BLEU分数':<20}: {results_model1['bleu']}")

    # 计算模型2的BLEU分数
    results_model2 = bleu.compute(predictions=pred2, references=references)
    print(f"\t{'模型2的BLEU分数':<20}: {results_model2['bleu']}")

    pbmBLUE_score = abs(results_model1['bleu'] - results_model2['bleu'])
    print(f"\t{'pbmBLUE_score':<20}: {pbmBLUE_score:.16f}")


# 指定GPU设备：
device_model1 = torch.device("cuda:0")  # 第x块GPU
device_model2 = torch.device("cuda:1")  # 第y块GPU

# 设置模型和输入
model_7b        = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b"
model_7b_Python = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b-Python"

model1, tokenizer1 = load_model(model_7b, device_model1)
model2, tokenizer2 = load_model(model_7b_Python, device_model2)


# 打开jsonl文件并遍历
file_path = '/newdisk/public/wws/humaneval-x-main/data/python/data/humaneval.jsonl'  # Dataset

# 加载 BLEU 指标
bleu = evaluate.load("bleu")
# 如果卡在load函数上, 将下面一行代码在bash上运行,再运行此.py
# export HF_ENDPOINT=https://hf-mirror.com

with jsonlines.open(file_path) as reader:
    for obj in reader:
        task_id = obj.get('task_id')
        prompt = obj.get('prompt')
        refer = obj.get('canonical_solution')
        # prompt = "def fibonacci("
        print(f"Task ID: {task_id}, Prompt: \n{prompt}")
               

        # 输出所有层的CCA分数后，生成Prompt的模型输出
        inputs = tokenizer1(prompt, return_tensors='pt').to(device_model1)
        output_model1 = model1.generate(**inputs, max_length=512)
        generated_text_model1 = tokenizer1.decode(output_model1[0], skip_special_tokens=True)
        
        inputs = tokenizer2(prompt, return_tensors='pt').to(device_model2)
        output_model2 = model2.generate(**inputs, max_length=512)
        generated_text_model2 = tokenizer2.decode(output_model2[0], skip_special_tokens=True)

        # 输出Prompt的模型生成结果
        print("\nGenerated text by CodeLlama-7b:\n")
        print(generated_text_model1)
        print("\nGenerated text by CodeLlama-7b-Python:\n")
        print(generated_text_model2)
        calculate_PBM_BLUE(generated_text_model1, generated_text_model2,refer)





