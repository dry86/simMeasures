import torch
from getHiddenStates import load_model, get_hidden_states
import numpy as np
import jsonlines
from example.Topology import *
import re

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

def mask_code_keywords(code: str) -> list:
    # 匹配 """ 注释的正则表达式
    comment_pattern = r'""".*?"""'

    # 找到所有 """ 注释内容
    comments = [match.span() for match in re.finditer(comment_pattern, code, re.DOTALL)]

    # 列表存储多次 mask 的结果
    masked_versions = []

    # 找到所有非注释区域的代码关键字和符号
    # 使用正则表达式捕获关键字和操作符，比如 return、标识符和操作符
    keyword_pattern = r'->|>=|<=|==|!=|\d+\.\d+|\d+|\b\w+\b|[%=+*/-]'  # 捕获字母、数字、和常见操作符

    # 提取所有关键字的位置
    mask_positions = [(match.group(), match.span()) for match in re.finditer(keyword_pattern, code)]

    # 过滤掉注释部分的关键字
    non_comment_positions = [(word, span) for word, span in mask_positions
                             if not any(start <= span[0] < end for start, end in comments)]

    # 对每个关键字进行 mask，生成不同的版本
    for word, (start, end) in non_comment_positions:
        # 生成代码副本并进行 mask
        new_code = list(code)  # 把代码转为字符列表，便于替换
        new_code[start:end] = '<mask>'  # 替换当前位置的关键字为 <mask>
        masked_versions.append("".join(new_code))

    return masked_versions

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
        ground_truth_code = prompt + refer
               
        masked_results = mask_code_keywords(ground_truth_code)
        # 输出每次 mask 掉后的结果
        for idx, masked_code in enumerate(masked_results, 1):
            print(f"Masked Version {idx}:\n{masked_code}\n")

            
            # 将 "<mask>" 替换为模型能够识别的特殊 token，比如 "<unk>" 或者 "<mask>"，具体取决于模型支持的 token
            masked_input = masked_code.replace("<mask>", "<unk>")  # 使用 <unk> 作为占位符
            inputs1 = tokenizer1(masked_input, return_tensors='pt').to(device_model1)
            # 找到 <mask> 的位置
            mask_token_index = torch.where(inputs1.input_ids == tokenizer1.convert_tokens_to_ids("<unk>"))[1]
            with torch.no_grad():
                output_model1 = model1(**inputs1)
            # 获取 mask 位置的 logits
            mask_logits = output_model1.logits[0, mask_token_index, :]
            # 获取预测的 token id
            predicted_token_id = torch.argmax(mask_logits, dim=-1).item()
            # 获取该 token id 对应的 token
            predicted_token = tokenizer1.decode(predicted_token_id)


            inputs = tokenizer2(masked_code, return_tensors='pt').to(device_model2)
            with torch.no_grad():
                output_model2 = model2(**inputs)
            logits_model2 = output_model2.logits
            probabilities_model2 = torch.softmax(logits_model2, dim=-1)







