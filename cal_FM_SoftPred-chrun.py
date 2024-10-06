import torch
from getHiddenStates import load_model, get_hidden_states
from transformers import AutoTokenizer, AutoModelForCausalLM
import jsonlines


# 定义模型加载函数
def load_model_and_tokenizer(model_path: str, device: torch.device):
    """
    加载预训练的模型和分词器，并将模型加载到指定设备上。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        pad_token_id=tokenizer.pad_token_id,
        torch_dtype=torch.float32
    ).to(device)
    return model, tokenizer

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
    jsd = (kl_O_O_bar + kl_O_prime_O_bar) / 2
    return jsd.mean().item()



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
    
    # 计算平均 Surrogate Churn
    m_schurn = torch.mean(churn) / 2
    
    return m_schurn.item()

def main(model_1, model_2, file_path, device1, device2):

    model1, tokenizer1 = load_model_and_tokenizer(model_1, device1)
    model2, tokenizer2 = load_model_and_tokenizer(model_2, device2)

    model1.eval()
    model2.eval()

    with jsonlines.open(file_path) as reader:
        for obj in reader:
            task_id = obj.get('task_id').split('/')[1]
            prompt = obj.get('prompt')
            
            # 生成填充内容
            logits1, prob1 = generate_soft_logits(model1, tokenizer1, prompt, device1)
            logits2, prob2 = generate_soft_logits(model2, tokenizer2, prompt, device2)

            score_schurn = cal_surrogate_churn(logits1, logits2.to(device1))
            score_jsd    = jensen_shannon_divergence(prob1, prob2.to(device1))
            print(f"\t score_schurn: {score_schurn}")
            print(f"\t score_jsd: {score_jsd}")
            # if int(task_id) > 3:
            #     break

if __name__ == "__main__":

    # 指定GPU设备：
    device_model1 = torch.device("cuda:0")  # 第x块GPU
    device_model2 = torch.device("cuda:1")  # 第y块GPU

    # device_model1 = 'cpu'
    # device_model2 = 'cpu'

    # 设置模型和输入
    model_1 = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b"
    model_2 = "/newdisk/public/wws/model_dir/codellama/CodeLlama-7b-Instruct-hf" # "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b-Python"

    # 打开jsonl文件并遍历
    file_path = '/newdisk/public/wws/humaneval-x-main/data/python/data/humaneval.jsonl'  # Dataset

    main(model_1, model_2, file_path, device_model1, device_model2)
