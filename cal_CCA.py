import torch
from getHiddenStates import load_model, tokens_get_hidden_states
import numpy as np
import jsonlines
from example import cca_core
from tqdm import tqdm
# 设置打印选项来显示所有元素
# torch.set_printoptions(threshold=torch.inf)

def cca_decomp(A, B, device):
    """Computes CCA vectors, correlations, and transformed matrices
    requires a < n and b < n
    Args:
        A: np.array of size a x n where a is the number of neurons and n is the dataset size
        B: np.array of size b x n where b is the number of neurons and n is the dataset size
    Returns:
        u: left singular vectors for the inner SVD problem
        s: canonical correlation coefficients
        vh: right singular vectors for the inner SVD problem
        transformed_a: canonical vectors for matrix A, a x n array
        transformed_b: canonical vectors for matrix B, b x n array
    """
    # Move A and B to the specified device
    A = A.to(device)
    B = B.to(device)
    assert A.shape[0] < A.shape[1]
    assert B.shape[0] < B.shape[1]

    evals_a, evecs_a = torch.linalg.eigh(A @ A.T)
    evals_a = (evals_a + torch.abs(evals_a)) / 2
    inv_a = torch.tensor([1 / torch.sqrt(x) if x > 0 else 0 for x in evals_a], device=device)

    evals_b, evecs_b = torch.linalg.eigh(B @ B.T)
    evals_b = (evals_b + torch.abs(evals_b)) / 2
    inv_b = torch.tensor([1 / torch.sqrt(x) if x > 0 else 0 for x in evals_b], device=device)

    cov_ab = A @ B.T

    temp = (
        (evecs_a @ torch.diag(inv_a) @ evecs_a.T)
        @ cov_ab
        @ (evecs_b @ torch.diag(inv_b) @ evecs_b.T)
    )

    try:
        u, s, vh = torch.linalg.svd(temp)
    except:
        u, s, vh = torch.linalg.svd(temp * 100)
        s = s / 100

    transformed_a = (u.T @ (evecs_a @ torch.diag(inv_a) @ evecs_a.T) @ A).T
    transformed_b = (vh @ (evecs_b @ torch.diag(inv_b) @ evecs_b.T) @ B).T
    return u, s, vh, transformed_a, transformed_b


def mean_sq_cca_corr(rho):
    """Compute mean squared CCA correlation
    :param rho: canonical correlation coefficients returned by cca_decomp(A,B)
    """
    # len(rho) is min(A.shape[0], B.shape[0])
    return torch.sum(rho * rho) / len(rho)


def mean_cca_corr(rho):
    """Compute mean CCA correlation
    :param rho: canonical correlation coefficients returned by cca_decomp(A,B)
    """
    # len(rho) is min(A.shape[0], B.shape[0])
    return torch.sum(rho) / len(rho)


def calculate_cca(acts1, acts2, idx):
    # acts1 = acts1.T # convert to neurons by datapoints
    # acts2 = acts2.T
    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")
    results = cca_core.get_cca_similarity(acts1, acts2, epsilon=1e-6, verbose=False)

    print(f"\tMean CCA similarity: {np.mean(results["cca_coef1"])}")

    svcca_res = cca_core.compute_svcca(acts1, acts2)
    print("\tSVCCA similarity: ", svcca_res)

    pwcca_mean, w, _ = cca_core.compute_pwcca(results, acts1, acts2)
    print("\tPWCCA similarity: ", pwcca_mean)

def pad_to_max_length(tensor_list, tokenizer):
    # 获取所有张量的最大长度
    max_length = max(tensor.shape[1] for tensor in tensor_list)
    
    # 对每个张量进行填充，使它们具有相同的长度
    padded_tensors = [torch.nn.functional.pad(tensor, (0, 0, 0, max_length - tensor.shape[1])) for tensor in tensor_list]   # , value=tokenizer.eos_token_id
    
    return torch.stack(padded_tensors)

def main(model1_path, model2_path, data_file_path, device1, device2):
    """主函数：加载模型、读取数据、计算CCA相似性"""
    # 加载模型和tokenizer
    model1, tokenizer1 = load_model(model1_path, device1)
    model2, tokenizer2 = load_model(model2_path, device2)
    
    # 使用 eos_token 作为 pad_token
    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer2.pad_token = tokenizer2.eos_token
    tokenizer1.padding_side = "right"   # 使用EOS时,向右填充
    tokenizer2.padding_side = "right"

    prompts = []
    padding_max_length = 0
    with jsonlines.open(data_file_path) as reader:
        for obj in reader:
            prompt = obj.get('prompt')
            prompt_length = tokenizer1(prompt, return_tensors='pt').input_ids.shape[1]
            padding_max_length = max(padding_max_length, prompt_length)

    # 读取数据文件
    with jsonlines.open(data_file_path) as reader:
        for obj in reader:
            task_id = obj.get('task_id')
            task_number = int(task_id.split('/')[-1])
            prompt = obj.get('prompt')
            print(f"Task ID: {task_id}")
            # print(f"Prompt: \n{prompt}")
            prompts.append(prompt)
            if task_number == 30:
                break

    # 获取所有 prompts 的输入张量，并进行填充
    inputs_model1 = tokenizer1(prompts, return_tensors='pt', padding='max_length', max_length=padding_max_length).to(device1)
    inputs_model2 = tokenizer2(prompts, return_tensors='pt', padding='max_length', max_length=padding_max_length).to(device2)

    # 获取隐藏层输出
    hidden_states_model1 = tokens_get_hidden_states(model1, inputs_model1, device1)
    hidden_states_model2 = tokens_get_hidden_states(model2, inputs_model2, device2)

    # 获取模型的总层数并计算每一层的CCA相关性得分
    num_layers = len(hidden_states_model1)
    for i in tqdm(range(num_layers)):

        # 先将每层所有数据的隐藏层激活拼接成三维矩阵 (batch_size, max_length, hidden_size)
        layer_activations_model1 = hidden_states_model1[i]  # 形状 (batch_size, max_length, hidden_size)
        layer_activations_model2 = hidden_states_model2[i]  # 形状 (batch_size, max_length, hidden_size)

        # 通过 view() 函数将其变成二维矩阵 (batch_size * max_length, hidden_size)
        acts1 = layer_activations_model1.view(-1, layer_activations_model1.shape[-1])
        acts2 = layer_activations_model2.view(-1, layer_activations_model2.shape[-1])

        acts1 = acts1.T # convert to neurons by datapoints
        acts2 = acts2.T

        u, s, vh, transformed_a, transformed_b = cca_decomp(acts1, acts2, device1)
        cal_mean_cca = mean_cca_corr(s)
        print(f"\tcal_mean_cca: {cal_mean_cca}")

        acts1_numpy = acts1.cpu().numpy()
        acts2_numpy = acts2.cpu().numpy()

        # 计算该层的CCA
        calculate_cca(acts1_numpy, acts2_numpy, i)

if __name__ == "__main__":
    # 指定GPU设备
    device_model1 = torch.device("cuda:2")
    device_model2 = torch.device("cuda:3")

    # 模型和数据路径
    model_7b = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b"
    model_7b_Python = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b-Python"
    
    data_file = "/newdisk/public/wws/humaneval-x-main/data/js/data/humaneval.jsonl"

    # 调用主函数
    main(model_7b, model_7b_Python, data_file, device_model1, device_model2)
            


