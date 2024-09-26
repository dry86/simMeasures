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


# 指定GPU设备：
device_model1 = torch.device("cuda:2")  # 第x块GPU
device_model2 = torch.device("cuda:3")  # 第y块GPU

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
            calculate_cca(acts1, acts2, i)


            

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





