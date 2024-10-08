import torch
import jsonlines
from utils import load_model_and_tokenizer


# 定义生成文本函数
def generate_soft_logits(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int = 10):

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    output = model(input_ids)
    # pos = input_ids.shape[1] - 1
    logits = output.logits
    # 获取 mask 的概率矩阵
    probabilities = torch.softmax(logits, dim=-1)
    # # 获取预测的 token id
    # predicted_token_id = torch.argmax(logits, dim=-1)
    # predicted_token_id = predicted_token_id.flatten().tolist()
    # # 获取该 token id 对应的 token
    # predicted_token = tokenizer.decode(predicted_token_id, skip_special_tokens=True)
    return logits, probabilities

def prediction_difference_torch(outputs, p=1):
    """
    计算多个模型的 Prediction Difference (PD)，使用 torch 实现。
    
    参数:
    outputs: 一个形状为 (M, N, C) 的 torch 张量，表示 M 个模型对 N 个样本的 C 个类别的输出概率或 logits。
    p: 范数的类型，默认 p=1（可以是 L1 范数，也可以是 L2 范数）。
    
    返回:
    PD: 预测差异的值
    """
    # outputs 的维度为 (M, N, C)，M 是模型数，N 是样本数，C 是类别数
    M, N, C = outputs.shape
    
    # 计算每个样本的平均输出 (N, C)
    avg_outputs = torch.mean(outputs, dim=0)
    
    # 初始化 PD
    PD = 0
    
    # 对每个样本计算所有模型的范数差异
    for i in range(N):
        for O in outputs:
            PD += torch.norm(O[i] - avg_outputs[i], p=p)
    
    # 最终除以模型数 M
    PD /= M
    
    return PD

def kl_divergence(p, q):
    """
    计算两个概率分布 p 和 q 之间的 KL 散度
    """
    p = p + 1e-10  # 避免数值误差
    q = q + 1e-10
    return torch.sum(p * torch.log(p / q), dim=1)

def jensen_shannon_divergence(O, O_prime):
    """
    计算两个分布 O 和 O_prime 的 Jensen-Shannon 散度
    O 和 O_prime 形状为 (N, C)，N 是样本数，C 是类别数
    """
    # 计算平均分布 O_bar
    O_bar = (O + O_prime) / 2
    
    # 计算 KL 散度
    kl_O_O_bar = kl_divergence(O, O_bar)
    kl_O_prime_O_bar = kl_divergence(O_prime, O_bar)
    
    # Jensen-Shannon Divergence
    jsd = (kl_O_O_bar + kl_O_prime_O_bar) 
    return jsd

def cal_surrogate_churn(O, O_prime, alpha=1):
    """
    计算 Surrogate Churn
    O 和 O_prime 是两个模型的logits输出，形状为(N, C)，其中N是样本数量，C是类别数量
    alpha: 幂指数参数，默认为1
    """
    # 确保两个张量的形状相同
    assert O.shape == O_prime.shape, "Logits tensors must have the same shape"
    
    # 计算每个样本的最大logits（用于归一化）
    O_max = torch.max(O, dim=1, keepdim=True).values
    O_prime_max = torch.max(O_prime, dim=1, keepdim=True).values
    
    # 将 logits 归一化
    O_normalized = O / O_max
    O_prime_normalized = O_prime / O_prime_max
    
    # 计算归一化 logits 的差值并取绝对值
    diff = torch.abs(O_normalized**alpha - O_prime_normalized**alpha)
    
    # 对每个样本求L1范数（按类别求和） 
    churn = torch.sum(diff, dim=1)
    
    # # 计算平均 Surrogate Churn
    # m_schurn = torch.mean(churn) / 2
    
    return churn

def cal_norm_of_soft_prediction_diff(O, O_prime):
    """
    计算两个输出O和O'之间的Norm of Soft Prediction Difference
    O和O'为两个模型的输出，形状为(N, C)，其中N是实例数，C是类数
    """

    # # 获取两者的形状，确保两个张量在形状上相同
    # min_length = min(O.shape[2], O_prime.shape[2])

    # # 截取logits的最后一维，使得它们形状一致
    # O_trimmed = O[:, :, :min_length]
    # O_prime_trimmed = O_prime[:, :, :min_length]

    # N = O.shape[0]
    # # 确保两个tensor在相同的设备上
    # if O_trimmed.device != O_prime_trimmed.device:
    #     O_prime_trimmed = O_prime_trimmed.to(O_trimmed.device)  # 将tensor2移动到tensor1所在的设备
    
    # 计算每个实例对应的欧几里得距离
    distances = torch.norm(O - O_prime, dim=0)
    
    # # 计算平均差异
    # m_pred_norm_diff = torch.sum(distances) / (2 * O_trimmed.shape[0])
    
    return distances


def main(model_1, model_2, file_path, device1, device2):

    model1, tokenizer1 = load_model_and_tokenizer(model_1, device1)
    model2, tokenizer2 = load_model_and_tokenizer(model_2, device2)

    model1.eval()
    model2.eval()

    score_normPD_log = 0
    score_normPD_prob = 0
    score_schurn = 0
    score_jsd = 0
    score_pd = 0
    N = 0 #  样本数
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            task_id = obj.get('task_id').split('/')[1]
            prompt = obj.get('prompt')
            
            # 生成logits, probability
            logits1, prob1 = generate_soft_logits(model1, tokenizer1, prompt, device1)
            logits2, prob2 = generate_soft_logits(model2, tokenizer2, prompt, device2)
            prob = torch.cat((prob1, prob2), dim=0)

            logits1 = logits1[0]
            logits2 = logits2[0]
            prob1 = prob1[0]
            prob2 = prob2[0]

            logits2 = logits2.to(device1)
            prob2 = prob2.to(device1)
            
            score_normPD_log += cal_norm_of_soft_prediction_diff(logits1, logits2)
            # score_normPD_prob += cal_norm_of_soft_prediction_diff(prob1, prob2)

            score_schurn += cal_surrogate_churn(logits1, logits2)
            score_jsd    += jensen_shannon_divergence(prob1, prob2)
            
            score_pd     += prediction_difference_torch(prob)

            N = int(task_id) + 1
            # if int(task_id) > 3:
            #     break
        res_normPD_log = score_normPD_log / (2 * N)
        res_normPD_prob = score_normPD_prob / (2 * N)
        res_schurn = score_schurn / (2 * N)
        res_jsd = score_jsd / (2 * N)
        res_pd = score_pd / N
        print(f"\t normPD_log: {res_normPD_log}")
        print(f"\t normPD_prob: {res_normPD_prob}")
        print(f"\t schurn: {res_schurn}")
        print(f"\t jsd: {res_jsd}")
        print(f"\t pd: {res_pd}")

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
