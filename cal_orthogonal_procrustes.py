import torch
from getHiddenStates import load_model, get_hidden_states
from scipy.linalg import orthogonal_procrustes
import jsonlines

def procrustes_2(A, B):
    """
    Computes Procrustes distance bewteen representations A and B
    for when |neurons| >> |examples| and A.T @ B too large to fit in memory.
    Based on:
         np.linalg.norm(A.T @ B, ord="nuc") == np.sum(np.sqrt(np.linalg.eig(((A @ A.T) @ (B @ B.T)))[0]))
    
    Parameters
    ----------
    A : examples x neurons
    B : examples x neurons

    Original Code
    -------------    
    nuc = np.linalg.norm(A @ B.T, ord="nuc")  # O(p * p * n)
    """

    A_sq_frob = torch.sum(A ** 2)
    B_sq_frob = torch.sum(B ** 2)
    nuc = torch.sum(torch.sqrt(torch.abs(torch.linalg.eig(((A @ A.T) @ (B @ B.T)))[0])))

    return A_sq_frob + B_sq_frob - 2 * nuc

def calculate_orthogonal_procrustes(matrix1, matrix2):
    """
    计算 orthogonal_procrustes :
    给定形状相同的矩阵A和B, 求得: 
    ①矩阵R, 该矩阵R是使得矩阵A经过变换后最接近矩阵B的正交矩阵; 
    ②scale, 等于矩阵A^T B的奇异值之和
    """
    # print(f"matrix1:\n {matrix1}")
    # print(f"matrix2:\n {matrix2}")
    R, sca = orthogonal_procrustes(matrix1, matrix2)

    return R, sca

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

        # 获取隐藏层矩阵
        hidden_states_model1 = get_hidden_states(model1, tokenizer1, prompt, device_model1)
        hidden_states_model2 = get_hidden_states(model2, tokenizer2, prompt, device_model2)

        # 获取模型的总层数
        num_layers = len(hidden_states_model1)
        # num_layers = 1

        for i in range(num_layers):
            R, sca = calculate_orthogonal_procrustes(hidden_states_model1[i].reshape(-1, hidden_states_model1[i].shape[-1]),
                                hidden_states_model2[i].reshape(-1, hidden_states_model2[i].shape[-1]))
            print(f"Layer {i} orthogonal_procrustes R: \n{R} , shape = {R.shape} \nscale: {sca}")

        
    

