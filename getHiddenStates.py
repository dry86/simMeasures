import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(device)  # 将模型移动到指定的GPU上
    return model, tokenizer

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