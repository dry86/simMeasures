import torch
from getHiddenStates import load_model, get_hidden_states
import numpy as np
import jsonlines
from example import cca_core



def calculate_cca(acts1, acts2, idx):
    
    print(f"Layer {idx}, shape: {acts1.shape}:")
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

    pwcca_mean, w, _ = cca_core.compute_pwcca(acts1, acts2, epsilon=1e-6)
    print("\tPWCCA similarity: ", pwcca_mean)


def main(model1_path, model2_path, data_file_path, device1, device2):
    """主函数：加载模型、读取数据、计算CCA相似性"""
    # 加载模型和tokenizer
    model1, tokenizer1 = load_model(model_7b, device_model1)
    model2, tokenizer2 = load_model(model_7b_Python, device_model2)

    # 读取数据文件
    with jsonlines.open(data_file_path) as reader:
        for obj in reader:
            task_id = obj.get('task_id')
            prompt = obj.get('prompt')
            print(f"Task ID: {task_id}, Prompt: \n{prompt}")

            # 获取隐藏层输出
            hidden_states_model1 = get_hidden_states(model1, tokenizer1, prompt, device1)
            hidden_states_model2 = get_hidden_states(model2, tokenizer2, prompt, device2)

            # 获取模型的总层数并计算每一层的CCA相关性得分
            num_layers = len(hidden_states_model1)
            for i in range(num_layers):
                acts1 = hidden_states_model1[i].reshape(-1, hidden_states_model1[i].shape[-1])
                acts2 = hidden_states_model2[i].reshape(-1, hidden_states_model2[i].shape[-1])
                calculate_cca(acts1, acts2, i)

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
            


