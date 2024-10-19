import os
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

def get_batch_number(filename, keyword):
    """从文件名中提取批次号，例如 'hsm1_batch_19.pt' 提取出 19"""
    return int(filename.split(f"{keyword}batch_")[-1].split('.pt')[0])

def concatenate_hidden_states(directory, keyword, device):
    """加载并拼接目录中以特定关键字开头的所有.pt文件的隐藏状态张量"""
    keyword = keyword + '_'
    hidden_states_list = []

    # 获取符合条件的文件并按批次号排序
    pt_files = [filename for filename in os.listdir(directory) if filename.startswith(keyword) and filename.endswith('.pt')]
    sorted_pt_files = sorted(pt_files, key=lambda x: get_batch_number(x, keyword))

    # 遍历排序后的文件名并加载张量
    for filename in sorted_pt_files:
        file_path = os.path.join(directory, filename)
        hidden_states = load_hidden_states(file_path, device)
        hidden_states_list.append(hidden_states)

    # 确保至少有一个张量可以拼接
    if not hidden_states_list:
        raise ValueError("没有找到符合条件的.pt文件！")

    # 检查所有张量的形状是否一致（序列长度和隐藏层维度）
    base_shape = hidden_states_list[0][0].shape[1:]  # 获取第一个张量的形状
    for hidden_states in hidden_states_list:
        assert hidden_states[0].shape[1:] == base_shape, "序列长度或隐藏层维度不一致！"

    num_layers = len(hidden_states_list[0])  # 33 层

    # 初始化列表用于保存每一层的拼接结果
    all_hidden_states = []

    # 遍历每一层（共33层），对每一层的batch维度（第0维）进行拼接
    for layer_idx in range(num_layers):
        # 从 hidden_states_list 中提取第 layer_idx 层，按第0维度拼接
        layer_activations = [hidden_states[layer_idx] for hidden_states in hidden_states_list]
        concatenated_layer = torch.cat(layer_activations, dim=0)  # 在 batch 维度拼接
        all_hidden_states.append(concatenated_layer)

    # # 将每层的拼接结果组成一个最终的张量，形状为 (num_layers, total_batch_size, seq_len, hidden_size)
    # all_hidden_states = torch.stack(all_hidden_states)

    # 将拼接后的张量移动到指定的设备
    # all_hidden_states = all_hidden_states.to(device)

    return all_hidden_states

def concatenate_last_layer_hidden_states(directory, keyword, device):
    """加载并拼接目录中以特定关键字开头的所有.pt文件的隐藏状态张量"""
    keyword = keyword + '_'
    hidden_states_last_layer_list = []

    # 获取符合条件的文件并按批次号排序
    pt_files = [filename for filename in os.listdir(directory) if filename.startswith(keyword) and filename.endswith('.pt')]
    sorted_pt_files = sorted(pt_files, key=lambda x: get_batch_number(x, keyword))

    # 遍历排序后的文件名并加载张量
    for filename in sorted_pt_files:
        file_path = os.path.join(directory, filename)
        hidden_states = load_hidden_states(file_path, device)
        hidden_states_last_layer_list.append(hidden_states[-1])    # only use the last layer

    # 确保至少有一个张量可以拼接
    if not hidden_states_last_layer_list:
        raise ValueError("没有找到符合条件的.pt文件！")

    # 按第0维度拼接
    layer_activations = [hidden_states for hidden_states in hidden_states_last_layer_list]
    concatenated_last_layer = torch.cat(layer_activations, dim=0)  # 在 batch 维度拼接

    return concatenated_last_layer

# 获取特定隐藏层输出
def get_special_hidden_states(model, tokenizer, input_text, layer_indices, device):
    inputs = tokenizer(input_text, return_tensors='pt').to(device)  # 将输入移动到指定的GPU上
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # 提取指定层的隐藏状态，并将其转移到CPU以便后续处理
    hidden_states = outputs.hidden_states
    return [hidden_states[i].detach().cpu().numpy() for i in layer_indices]

# 获取特定隐藏层输出
def get_last_layer_hidden_states(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # 提取所有层的隐藏状态
    hidden_states = outputs.hidden_states
    
    # 提取最后一层隐藏层的隐藏状态
    last_hidden_state = hidden_states[-1]
    
    return outputs, last_hidden_state

# 获取隐藏层输出
def get_hidden_states(model, tokenizer, input_text, device):
    inputs = tokenizer(input_text, return_tensors='pt').to(device)  # 将输入移动到指定的GPU上
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # 提取所有层的隐藏状态，并将其转移到CPU以便后续处理
    hidden_states = outputs.hidden_states
    return hidden_states
    # return [layer_hidden_state.detach().cpu().numpy() for layer_hidden_state in hidden_states]

def tokens_get_hidden_states(model, inputs, device):
    # 确保模型处于评估模式
    model.eval()
    
    # 禁用梯度计算，节省内存和加速计算
    with torch.no_grad():
        # 前向传播，获取模型输出
        outputs = model(**inputs, output_hidden_states=True)
        
        # 获取所有层的隐藏状态，outputs.hidden_states 是一个元组，包含每一层的输出
        hidden_states = outputs.hidden_states

        return outputs, hidden_states


# 获取logits输出
def get_logits(model, tokenizer, input_text, device):
    inputs = tokenizer(input_text, return_tensors='pt').to(device)  # 将输入移动到指定的GPU上
    outputs = model(**inputs)

    return outputs.logits