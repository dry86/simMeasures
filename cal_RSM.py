import torch
from getHiddenStates import load_model, get_hidden_states
from sklearn.cross_decomposition import CCA
import numpy as np
import jsonlines
from example.RSM import cal_RSM_Norm_Difference, cal_RSA, cal_cka, cal_distance_correlation, cal_bures_similarity, cal_eigenspace_overlap_score, cal_gulp_measure, cal_riemannian_distance

def calculate_RSM(acts1, acts2, idx):
    print(f"Layer {idx}, shape: {acts1.shape}:")
    # 计算rsm相似度
    similarity_rsm = cal_RSM_Norm_Difference(acts1, acts2)
    print(f"\t{'Representational Similarity (Frobenius norm difference)':<60}: {similarity_rsm:.16f}")

    # 计算rsa相似度
    similarity_rsa = cal_RSA(acts1, acts2)
    print(f"\t{'Representational Similarity (Pearson correlation)':<60}: {similarity_rsa:.16f}")

    # 计算cka相似度
    similarity_cka = cal_cka(acts1, acts2)
    print(f"\t{'CKA Similarity':<60}: {similarity_cka:.16f}")

    # 计算dCor距离相关性
    dcor_score = cal_distance_correlation(acts1, acts2)
    print(f"\t{'Distance Correlation':<60}: {dcor_score:.16f}")

    # 计算 Normalized Bures Similarity
    nbs_score = cal_bures_similarity(acts1, acts2)
    print(f"\t{'Normalized Bures Similarity':<60}: {np.real(nbs_score)}")

    # 计算 Eigenspace Overlap Score
    eos_score = cal_eigenspace_overlap_score(acts1, acts2)
    print(f"\t{'Eigenspace Overlap Score (Normalized)':<60}: {eos_score}")

    # 计算 Unified Linear Probing (GULP).
    glup_score = cal_gulp_measure(acts1, acts2)
    print(f"\t{'Unified Linear Probing (GULP)':<60}: {glup_score}")

    # 计算 Riemannian Distance
    riemann_dist = cal_riemannian_distance(acts1, acts2)
    print(f"\t{'Riemannian Distance':<60}: {riemann_dist}")


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
        # prompt = "def fibonacci("
        print(f"Task ID: {task_id}, Prompt: \n{prompt}")
        
        # layer_indices = [1, -2]  # 倒数第二层和第二层

        # 获取隐藏层矩阵
        hidden_states_model1 = get_hidden_states(model1, tokenizer1, prompt, device_model1)
        hidden_states_model2 = get_hidden_states(model2, tokenizer2, prompt, device_model2)

        
        # 获取模型的总层数
        num_layers = len(hidden_states_model1)

        # 获取每一层的CCA相关性得分
        
        for i in range(num_layers):
            acts1 = hidden_states_model1[i].reshape(-1, hidden_states_model1[i].shape[-1])
            acts2 = hidden_states_model2[i].reshape(-1, hidden_states_model2[i].shape[-1])
            # print(f"hidden layer shape: {acts1.shape}")
            calculate_RSM(acts1, acts2, i)
            

        # 输出所有层的CCA分数后，生成Prompt的模型输出
        inputs = tokenizer1(prompt, return_tensors='pt').to(device_model1)
        output_model1 = model1.generate(**inputs, max_length=512)
        generated_text_model1 = tokenizer1.decode(output_model1[0], skip_special_tokens=True)
        
        inputs = tokenizer2(prompt, return_tensors='pt').to(device_model2)
        output_model2 = model2.generate(**inputs, max_length=512)
        generated_text_model2 = tokenizer2.decode(output_model2[0], skip_special_tokens=True)

        # 输出Prompt的模型生成结果
        print("\nGenerated text by CodeLlama-7b:\n")
        print(generated_text_model1)
        print("\nGenerated text by CodeLlama-7b-Python:\n")
        print(generated_text_model2)