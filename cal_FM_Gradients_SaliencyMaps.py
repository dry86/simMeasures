import torch
from getHiddenStates import load_model, get_hidden_states
import numpy as np
import jsonlines
from scipy.spatial.distance import cosine
import math
# 设置打印选项来显示所有元素
# torch.set_printoptions(threshold=torch.inf)

def compute_saliency_map(model, inputs):
    """
    计算每一层隐藏层的梯度。
    
    Args:
        model: 预训练语言模型，例如 codeLlama-7b。
        inputs: 模型的输入数据，包含 input_ids 和 attention_mask。
    
    Returns:
        saliency_maps: 每一层隐藏层的梯度映射。
    """
    hidden_states = []

    # 定义一个 hook 函数来捕捉每一层的输出
    def hook_fn(module, input, output):
        hidden_states.append(output)
        # 检查 output 是不是一个元组，如果是，对每个元素调用 retain_grad
        if isinstance(output, tuple):
            for out in output:
                if isinstance(out, torch.Tensor):
                    out.retain_grad()  # 只对张量调用 retain_grad
        else:
            output.retain_grad()  # 如果不是元组，直接调用 retain_grad

    # 注册 hook 到所有隐藏层
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(hook_fn))

    # 确保输入中有 'labels' 字段
    inputs['labels'] = inputs['input_ids']  # 或者你希望计算损失的其他标签

    # 前向传播，获取损失
    outputs = model(**inputs)
    loss = outputs.loss

    # 反向传播，计算梯度
    loss.backward()

    saliency_maps = []

    # 计算每一层的梯度
    for hidden_state in hidden_states:
        # 如果 all_tokens=True，计算所有 token 的梯度
        saliency_map = hidden_state[0].grad
        # 确保张量在 CPU 上并转换为 NumPy 数组
        if saliency_map is not None:  # 检查是否存在梯度
            saliency_map = saliency_map.cpu().numpy().astype(np.float64)  # 转换为 NumPy 数组   
        saliency_maps.append(saliency_map)

    # 移除 hooks
    for hook in hooks:
        hook.remove()

    return saliency_maps

precision_count = 0

def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度"""
    
    # 将 PyTorch tensor 转换为 NumPy 数组（如果是 tensor）
    v1 = v1.cpu().numpy() if isinstance(v1, torch.Tensor) else v1
    v2 = v2.cpu().numpy() if isinstance(v2, torch.Tensor) else v2
    
    # 计算向量的点积和范数
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)


    if (1e-10 < norm_v1 < 1e-8) or (1e-10 < norm_v2 < 1e-8):
        print("find norm_v1_v2 1e-8 -10")

    
    # # 检查是否有零向量，防止除以零的情况
    if norm_v1 < 1e-10 or norm_v2 < 1e-10:
        global precision_count
        precision_count = precision_count + 1
        # print("find norm_v1_v2 1e-10")
        return 1.0  # 或者返回 np.nan，根据需求

    # 手动计算余弦相似度
    cos_sim = dot_product / (norm_v1 * norm_v2)
    
    # 检查 cos_sim 是否为有效值（非 NaN 或无穷大）
    if np.isnan(cos_sim):
        print(f"find NaN")
        return 0.0  # 根据需求处理
    if np.isinf(cos_sim):
        print("find inf")
        return 0.0  # 根据需求处理
    
    # 将 cos_sim 限制在 [-1, 1] 的范围内
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    return cos_sim

def compute_layer_token_similarity(grad1, grad2):
    """
    计算两个隐藏层的每个 token 梯度的余弦相似度。
    
    Args:
        grad1: 模型1某一层的所有 token 梯度, 形状为 (batch_size, seq_len, hidden_size)。
        grad2: 模型2某一层的所有 token 梯度, 形状为 (batch_size, seq_len, hidden_size)。
    Returns:
        similarities: 每个 token 的余弦相似度, 形状为 (seq_len,)。
    """
    batch_size, seq_len, hidden_size = grad1.shape
    similarities = []
    
    # 对每个 token 计算余弦相似度
    for token_idx in range(seq_len):
        v1 = grad1[:, token_idx, :].flatten()  # 取出模型1在该token位置的梯度
        v2 = grad2[:, token_idx, :].flatten()  # 取出模型2在该token位置的梯度
        # if token_idx == 132:
        #     print("132")
        sim = cosine_similarity(v1, v2)
        similarities.append(sim)
    
    return similarities

def compute_similarity(model1, model2, inputs1, inputs2):
    """
    计算两个模型在每个隐藏层上所有 token 的梯度相似度。
    
    Args:
        model1: 模型1。
        model2: 模型2。
        inputs1: 模型1的输入。
        inputs2: 模型2的输入。
    
    Returns:
        layer_similarities: 每层所有 token 相似度的平均值, 形状为 (num_layers,)。
    """
    # 获取两个模型的梯度
    grad1 = compute_saliency_map(model1, inputs1)
    grad2 = compute_saliency_map(model2, inputs2)

    layer_similarities = []
    
    # 对每一层的所有 token 梯度进行比较
    for layer_idx in range(len(grad1)):
        # 对应层的所有 token 梯度
        layer_grad1 = grad1[layer_idx]
        layer_grad2 = grad2[layer_idx]
        
        # 计算该层中所有 token 的相似度
        token_similarities = compute_layer_token_similarity(layer_grad1, layer_grad2)
        # 过滤掉 NaN 值
        filtered_token_similarities = [sim for sim in token_similarities if not math.isnan(sim)]
        # 计算该层所有 token 相似度的平均值
        avg_token_similarity = sum(token_similarities) / len(token_similarities)
        layer_similarities.append(avg_token_similarity)
    
    return layer_similarities

def compute_average_similarity(model1, model2, inputs1, inputs2):
    """
    计算两个模型所有层的平均相似度。
    
    Args:
        model1: 模型1。
        model2: 模型2。
        inputs1: 模型1的输入。
        inputs2: 模型2的输入。
    
    Returns:
        avg_similarity: 所有层相似度的平均值。
    """
    layer_similarities = compute_similarity(model1, model2, inputs1, inputs2)
    
    # 计算所有层相似度的平均值
    avg_similarity = sum(layer_similarities) / len(layer_similarities)
    
    return avg_similarity


def main(model_1, model_2, file_path, device1, device2):

    model1, tokenizer1 = load_model(model_1, device1)
    model2, tokenizer2 = load_model(model_2, device2)

    model1.half()
    model2.half()

    # 切换到evaluation模式
    # model1.eval()
    # model2.eval()

    with jsonlines.open(file_path) as reader:
        for obj in reader:
            task_id = obj.get('task_id')
            prompt = obj.get('prompt')
            # refer = obj.get('canonical_solution')
            # prompt = "def fibonacci("
            print(f"Task ID: {task_id}")
            # print(f"Prompt: \n{prompt}")

            inputs1 = tokenizer1(prompt, return_tensors='pt').to(device1)

            inputs2 = tokenizer2(prompt, return_tensors='pt').to(device2)

            avg_similarity = compute_average_similarity(model1, model2, inputs1, inputs2)
            print(avg_similarity)


            
    print(f"global precision count:{precision_count}")

if __name__ == "__main__":

    # 指定GPU设备：
    device_model1 = torch.device("cuda:0")  # 第x块GPU
    device_model2 = torch.device("cuda:1")  # 第y块GPU

    device_model1 = 'cpu'
    device_model2 = 'cpu'

    # 设置模型和输入
    model_1 = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b"
    model_2 = "/newdisk/public/wws/model_dir/codellama/CodeLlama-7b-Instruct-hf" # "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b-Python"

    # 打开jsonl文件并遍历
    file_path = '/newdisk/public/wws/humaneval-x-main/data/python/data/humaneval.jsonl'  # Dataset

    main(model_1, model_2, file_path, device_model1, device_model2)