import torch
from getHiddenStates import load_model, get_hidden_states
import numpy as np
import jsonlines
from example.Topology import *

def cal_norm_of_soft_prediction_diff(O, O_prime):
    """
    计算两个输出O和O'之间的Norm of Soft Prediction Difference
    O和O'为两个模型的输出，形状为(N, C)，其中N是实例数，C是类数
    """

    # 获取两者的形状，确保两个张量在形状上相同
    min_length = min(O.shape[2], O_prime.shape[2])

    # 截取logits的最后一维，使得它们形状一致
    O_trimmed = O[:, :, :min_length].detach().cpu().numpy()
    O_prime_trimmed = O_prime[:, :, :min_length].detach().cpu().numpy()

    # N = O.shape[0]
    # # 确保两个tensor在相同的设备上
    # if O_trimmed.device != O_prime_trimmed.device:
    #     O_prime_trimmed = O_prime_trimmed.to(O_trimmed.device)  # 将tensor2移动到tensor1所在的设备
    
    # 计算每个实例对应的欧几里得距离
    distances = np.linalg.norm(O_trimmed - O_prime_trimmed, axis=2)
    
    # 计算平均差异
    m_pred_norm_diff = np.sum(distances) / (2 * O_trimmed.shape[0])
    
    return m_pred_norm_diff

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
        refer = obj.get('canonical_solution')
        # prompt = "def fibonacci("
        print(f"Task ID: {task_id}, Prompt: \n{prompt}")
               
        # 生成模型的logits, probabilities输出
        inputs = tokenizer1(prompt, return_tensors='pt').to(device_model1)
        output_model1 = model1(**inputs)
        logits_model1 = output_model1.logits
        # 使用softmax将logits转换为概率分布
        probabilities_model1 = torch.softmax(logits_model1, dim=-1)
        # generated_text_model1 = tokenizer1.decode(output_model1[0], skip_special_tokens=True)
        
        inputs = tokenizer2(prompt, return_tensors='pt').to(device_model2)
        output_model2 = model2(**inputs)
        logits_model2 = output_model2.logits
        probabilities_model2 = torch.softmax(logits_model2, dim=-1)
        # generated_text_model2 = tokenizer2.decode(output_model2[0], skip_special_tokens=True)

        logits_norm_diff = cal_norm_of_soft_prediction_diff(logits_model1, logits_model2)
        print(f"Logits Norm of Soft Prediction Difference: {logits_norm_diff}")

        probabilities_norm_diff = cal_norm_of_soft_prediction_diff(probabilities_model1, probabilities_model2)
        print(f"Probabilities Norm of Soft Prediction Difference: {probabilities_norm_diff}")




