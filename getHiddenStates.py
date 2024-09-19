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
    outputs = model(**inputs)
    
    # 提取所有层的隐藏状态，并将其转移到CPU以便后续处理
    hidden_states = outputs.hidden_states
    return [layer_hidden_state.detach().cpu().numpy() for layer_hidden_state in hidden_states]

# 获取logits输出
def get_logits(model, tokenizer, input_text, device):
    inputs = tokenizer(input_text, return_tensors='pt').to(device)  # 将输入移动到指定的GPU上
    outputs = model(**inputs)

    return outputs.logits