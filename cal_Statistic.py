import torch
from getHiddenStates import load_model, get_hidden_states
import numpy as np
import jsonlines
from example.Statistic_matrix import *


def calculate_Stat(acts1, acts2):
    
    print(f"All layer, shape: {acts1.shape}:")
    
    # 计算 mag, var_mag
    mag_1 = mean_magnitude(acts1)
    var_mag_1 = magnitude_variance(acts1)

    print(f"\t{'Magnitude acts1':<30}: {mag_1}")
    print(f"\t{'Magnitude Variance acts1':<30}: {var_mag_1}")

    mag_2 = mean_magnitude(acts2)
    var_mag_2 = magnitude_variance(acts2)

    print(f"\t{'Magnitude acts2':<30}: {mag_2}")
    print(f"\t{'Magnitude Variance acts2':<30}: {var_mag_2}")

    # 计算 conc, var_conc
    mean_conc_1 = mean_concentricity(acts1)
    var_conc_1 = concentricity_variance(acts1)

    print(f"\t{'Mean Concentricity acts1':<30}: {mean_conc_1}")
    print(f"\t{'Concentricity Variance acts1':<30}: {var_conc_1}")

    mean_conc_2 = mean_concentricity(acts2)
    var_conc_2 = concentricity_variance(acts2)

    print(f"\t{'Mean Concentricity acts2':<30}: {mean_conc_2}")
    print(f"\t{'Concentricity Variance acts2':<30}: {var_conc_2}")


    # 计算 uniformity
    uniformity_value_1 = uniformity_euclidean(acts1)
    print(f"\t{'Uniformity euclidean acts1':<30}: {uniformity_value_1}")

    uniformity_value_2 = uniformity_euclidean(acts2)
    print(f"\t{'Uniformity euclidean acts2':<30}: {uniformity_value_2}")

    uniformity_value_1 = uniformity_inner(acts1)
    print(f"\t{'Uniformity inner acts1':<30}: {uniformity_value_1}")

    uniformity_value_2 = uniformity_inner(acts2)
    print(f"\t{'Uniformity inner acts2':<30}: {uniformity_value_2}")

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

with jsonlines.open(file_path) as reader:
    for obj in reader:
        task_id = obj.get('task_id')
        prompt = obj.get('prompt')
        # prompt = "def fibonacci("
        print(f"Task ID: {task_id}, Prompt: \n{prompt}")
        
        # layer_indices = [1, -2]  # 倒数第二层和第二层

        # 获取隐藏层矩阵
        hidden_states_model1 = get_hidden_states(model1, tokenizer1, prompt, device_model1)
        hidden_states_model2 = get_hidden_states(model2, tokenizer2, prompt, device_model2)

        
        # 获取模型的总层数
        # num_layers = len(hidden_states_model1)
        
        # for i in range(num_layers):
        acts1 = hidden_states_model1.reshape(-1, hidden_states_model1.shape[-1])
        acts2 = hidden_states_model2.reshape(-1, hidden_states_model2.shape[-1])
        # print(f"hidden layer shape: {acts1.shape}")
        calculate_Stat(acts1, acts2)
            

        
        # inputs = tokenizer1(prompt, return_tensors='pt').to(device_model1)
        # output_model1 = model1.generate(**inputs, max_length=512)
        # generated_text_model1 = tokenizer1.decode(output_model1[0], skip_special_tokens=True)
        
        # inputs = tokenizer2(prompt, return_tensors='pt').to(device_model2)
        # output_model2 = model2.generate(**inputs, max_length=512)
        # generated_text_model2 = tokenizer2.decode(output_model2[0], skip_special_tokens=True)

        # # 输出Prompt的模型生成结果
        # print("\nGenerated text by CodeLlama-7b:\n")
        # print(generated_text_model1)
        # print("\nGenerated text by CodeLlama-7b-Python:\n")
        # print(generated_text_model2)





