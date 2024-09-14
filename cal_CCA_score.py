import torch
from getHiddenStates import load_model, get_hidden_states
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import jsonlines


## 计算CCA
def old_calculate_cca(matrix1, matrix2):
    # 获取矩阵的最小列数，作为 n_components 的上限
    min_components = min(matrix1.shape[1], matrix2.shape[1])
    
    # 确保 n_components 不超过最小列数
    cca = CCA(n_components=min(min_components, 6))  # 设置n_components为不超过最小值  n_components的作用
    cca.fit(matrix1, matrix2)
    
    # 计算投影后的矩阵
    # x_c, y_c = cca.transform(matrix1, matrix2)
    
    # 使用 score() 计算相关性得分
    score = cca.score(matrix1, matrix2)
    
    return score

def calculate_cca(matrix1, matrix2, n_components=50):
    # # 标准化输入数据，防止数值不稳定
    # scaler = StandardScaler()
    # matrix1 = scaler.fit_transform(matrix1)
    # matrix2 = scaler.fit_transform(matrix2)
    
    # # 可选：对数据进行降维以确保 SVD 收敛
    # pca = PCA(n_components=min(matrix1.shape[0], matrix2.shape[0],matrix1.shape[1], matrix2.shape[1], 100))  # 选择降维目标
    # matrix1 = pca.fit_transform(matrix1)
    # matrix2 = pca.fit_transform(matrix2)

    # 创建 CCA 对象，设置 n_components
    cca = CCA(n_components=n_components, max_iter=5000)
    # 进行 CCA 拟合
    cca.fit(matrix1, matrix2)
    
    # 返回拟合得分
    score = cca.score(matrix1, matrix2)
    return score

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
        cca_scores = []
        for i in range(num_layers):
            score = calculate_cca(hidden_states_model1[i].reshape(-1, hidden_states_model1[i].shape[-1]),
                                hidden_states_model2[i].reshape(-1, hidden_states_model2[i].shape[-1]))
            print(f"\tLayer {i} CCA score: {score}")

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





