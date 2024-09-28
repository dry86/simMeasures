import torch
from getHiddenStates import load_model, get_hidden_states
import numpy as np
import jsonlines
from example import cca_core
# 设置打印选项来显示所有元素
# torch.set_printoptions(threshold=torch.inf)

def cca(features_x, features_y):
    """Compute the mean squared CCA correlation (R^2_{CCA}).

    Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.

    Returns:
    The mean squared CCA correlations between X and Y.
    """
    qx, _ = np.linalg.qr(features_x)  # Or use SVD with full_matrices=False.
    qy, _ = np.linalg.qr(features_y)
    return np.linalg.norm(qx.T.dot(qy)) ** 2 / min(
        features_x.shape[1], features_y.shape[1])

def calculate_cca(acts1, acts2, idx):
    # acts1 = acts1.T # convert to neurons by datapoints
    # acts2 = acts2.T
    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")
    results = cca_core.get_cca_similarity(acts1, acts2, epsilon=1e-6, verbose=False)

    print(f"\tMean CCA similarity: {np.mean(results["cca_coef1"])}")

    # Results using SVCCA keeping 20 dims

    # Mean subtract activations
    cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    svacts1 = np.dot(s1[:20]*np.eye(20), V1[:20])
    svacts2 = np.dot(s2[:20]*np.eye(20), V2[:20])

    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-6, verbose=False)

    print("\tSVCCA similarity: ", np.mean(svcca_results["cca_coef1"]))

    pwcca_mean, w, _ = cca_core.compute_pwcca(results, acts1, acts2)
    print("\tPWCCA similarity: ", pwcca_mean)

def pad_to_max_length(tensor_list):
    # 获取所有张量的最大长度
    max_length = max(tensor.shape[1] for tensor in tensor_list)
    
    # 对每个张量进行填充，使它们具有相同的长度
    padded_tensors = [torch.nn.functional.pad(tensor, (0, 0, 0, max_length - tensor.shape[1])) for tensor in tensor_list]
    
    return torch.stack(padded_tensors)

def main(model1_path, model2_path, data_file_path, device1, device2):
    """主函数：加载模型、读取数据、计算CCA相似性"""
    # 加载模型和tokenizer
    model1, tokenizer1 = load_model(model1_path, device1)
    model2, tokenizer2 = load_model(model2_path, device2)

    # 用于存储所有数据每一层的隐藏层
    all_hidden_states_model1 = []
    all_hidden_states_model2 = []

    # 读取数据文件
    with jsonlines.open(data_file_path) as reader:
        for obj in reader:
            task_id = obj.get('task_id')
            task_number = int(task_id.split('/')[-1])
            prompt = obj.get('prompt')
            print(f"Task ID: {task_id}")
            # print(f"Prompt: \n{prompt}")
            if task_number == 40:
                break

            # 获取隐藏层输出
            hidden_states_model1 = get_hidden_states(model1, tokenizer1, prompt, device1)
            hidden_states_model2 = get_hidden_states(model2, tokenizer2, prompt, device2)

            # 累加每一层的隐藏层激活
            if not all_hidden_states_model1:
                all_hidden_states_model1 = [[layer] for layer in hidden_states_model1]
                all_hidden_states_model2 = [[layer] for layer in hidden_states_model2]
            else:
                for i in range(len(hidden_states_model1)):
                    all_hidden_states_model1[i].append(hidden_states_model1[i])
                    all_hidden_states_model2[i].append(hidden_states_model2[i])

    # 获取模型的总层数并计算每一层的CCA相关性得分
    num_layers = len(all_hidden_states_model1)
    for i in range(num_layers):
        # 先将每层所有数据的隐藏层激活拼接成三维矩阵并进行填充
        layer_activations_model1 = pad_to_max_length(all_hidden_states_model1[i])
        layer_activations_model2 = pad_to_max_length(all_hidden_states_model2[i])

        # 通过 view() 函数将其变成二维矩阵 (164 * max_seq_length, 4096)
        acts1 = layer_activations_model1.view(-1, layer_activations_model1.shape[-1])
        acts2 = layer_activations_model2.view(-1, layer_activations_model2.shape[-1])
        acts1 = acts1.T
        acts2 = acts2.T
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
            


