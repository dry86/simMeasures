import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(device)  # 将模型移动到指定的GPU上
    return model, tokenizer

def load_hidden_states(file_path, device):
    """加载隐藏层输出张量并指定加载的设备"""
    return torch.load(file_path, map_location=device, weights_only=True)

def load_concat_hidden_states(file_path, device):
    data = torch.load(file_path, map_location=device, weights_only=True)
    if isinstance(data, tuple):
        # 假设张量是元组的第一个元素
        return data[0]  
    return data

def extract_number(filename):
    """从文件名中提取数字"""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')  # 如果没有数字，放到最后

def concatenate_hidden_states(directory, keyword, device):
    """加载并拼接目录中以特定关键字开头的所有.pt文件的隐藏状态张量"""
    hidden_states_list = []

    # 找到所有符合条件的文件名
    pt_files = [filename for filename in os.listdir(directory) if filename.startswith(keyword) and filename.endswith('.pt')]

    # 按文件名中的数字进行排序
    pt_files = sorted(pt_files, key=extract_number)

    # 遍历排序后的文件名并加载张量
    for filename in pt_files:
        file_path = os.path.join(directory, filename)
        hidden_states = load_concat_hidden_states(file_path, device)
        hidden_states_list.append(hidden_states)

    # 确保至少有一个张量可以拼接
    if not hidden_states_list:
        raise ValueError("没有找到符合条件的.pt文件！")

    # 检查所有张量的形状是否一致（序列长度和隐藏层维度）
    base_shape = hidden_states_list[0].shape[1:]  # 获取第一个张量的形状
    for hidden_states in hidden_states_list:
        assert hidden_states.shape[1:] == base_shape, "序列长度或隐藏层维度不一致！"

    # 拼接所有张量
    combined_hidden_states = torch.cat(hidden_states_list, dim=0)  # 按 batch 维度拼接

    # 将拼接后的张量移动到指定的设备
    combined_hidden_states = combined_hidden_states.to(device)

    return combined_hidden_states

# 获取特定隐藏层输出
def get_special_hidden_states(model, tokenizer, input_text, layer_indices, device):
    inputs = tokenizer(input_text, return_tensors='pt').to(device)  # 将输入移动到指定的GPU上
    outputs = model(**inputs)
    
    # 提取指定层的隐藏状态，并将其转移到CPU以便后续处理
    hidden_states = outputs.hidden_states
    return [hidden_states[i].detach().cpu().numpy() for i in layer_indices]

# 获取隐藏层输出
def get_hidden_states(model, tokenizer, input_text, device):
    inputs = tokenizer(input_text, return_tensors='pt').to(device)  # 将输入移动到指定的GPU上
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 提取所有层的隐藏状态，并将其转移到CPU以便后续处理
    hidden_states = outputs.hidden_states
    return hidden_states
    # return [layer_hidden_state.detach().cpu().numpy() for layer_hidden_state in hidden_states]

def tokens_get_hidden_states(model, inputs, device):
    """
    获取模型的所有隐藏层输出
    参数:
    - model: 已加载的模型
    - inputs: tokenizer 编码后的输入
    - device: 模型运行的设备 (e.g., 'cuda' or 'cpu')
    
    返回:
    - 一个包含所有隐藏层输出的列表，每个元素对应一层的隐藏状态
    """
    # 确保模型处于评估模式
    model.eval()
    
    # 禁用梯度计算，节省内存和加速计算
    with torch.no_grad():
        # 前向传播，获取模型输出
        outputs = model(**inputs, output_hidden_states=True)
        
        # 获取所有层的隐藏状态，outputs.hidden_states 是一个元组，包含每一层的输出
        hidden_states = outputs.hidden_states
        
        # 将每一层的输出移到指定设备上
        # hidden_states = [layer_hidden_state.to(device) for layer_hidden_state in hidden_states]
        
        return hidden_states


# 获取logits输出
def get_logits(model, tokenizer, input_text, device):
    inputs = tokenizer(input_text, return_tensors='pt').to(device)  # 将输入移动到指定的GPU上
    outputs = model(**inputs)

    return outputs.logits