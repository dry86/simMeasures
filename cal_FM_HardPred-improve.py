import torch
from getHiddenStates import load_model, get_hidden_states
from transformers import AutoTokenizer, AutoModelForCausalLM
import jsonlines
import re
from tqdm import tqdm

# 使用正则表达式捕获关键字和操作符，比如 标识符, 操作符, 数字
keyword_pattern = r'->|>=|<=|==|!=|\d+\.\d+|\d+|\b\w+\b|[%=+*/-]'  # 捕获字母、数字、和常见操作符

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

def mask_code_keywords(code: str) -> list:
    # 匹配 """ 注释的正则表达式
    comment_pattern = r'""".*?"""'

    # 找到所有 """ 注释内容
    comments = [match.span() for match in re.finditer(comment_pattern, code, re.DOTALL)]

    # 列表存储多次 mask 的结果
    masked_versions = []

    # 列表存储每次被 mask 掉的关键字
    ground_labels = []

    # 找到所有非注释区域的代码关键字和符号
    # 提取所有关键字的位置
    mask_positions = [(match.group(), match.span()) for match in re.finditer(keyword_pattern, code)]

    # 过滤掉注释部分的关键字
    non_comment_positions = [(word, span) for word, span in mask_positions
                             if not any(start <= span[0] < end for start, end in comments)]

    # 对每个关键字进行 mask，生成不同的版本
    for word, (start, end) in non_comment_positions:
        # 生成代码副本并进行 mask
        new_code = list(code)  # 把代码转为字符列表，便于替换
        new_code[start:end] = '<FILL_ME>'  # 替换当前位置的关键字为 <FILL_ME>
        masked_versions.append("".join(new_code))

        # 保存被 mask 掉的关键字
        ground_labels.append(word)

    return masked_versions, ground_labels

def is_multi_token(tokenizer, text: str):
    """
    判断给定的文本是否由多个 token id 组成。
    
    Args:
    - tokenizer: 使用的分词器。
    - text: 需要检查的文本。
    
    Returns:
    - bool: 如果文本由多个 token 组成，则返回 True；否则返回 False。
    - token_id: 对应的 token id。
    """
    # 获取token_ids
    token_id = tokenizer.convert_tokens_to_ids(text)

    # 如果 token_id == 0，说明该文本不在词汇表中
    if token_id == 0:
        return True
    else:
        return False

# 定义生成文本函数
def generate_outputs(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int = 6):
    """
    基于给定的 prompt 生成新的文本，并返回生成过程中每个token的logits, probabilities, token id, text。
    """
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]
    attention_mask = inputs["attention_mask"]

    generated_text = prompt
    logits_first_token = []

    # 逐步生成新token
    for i in range(max_new_tokens):
        # 前向传递获取logits
        with torch.no_grad():
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask)
            logits = outputs.logits
        
        # 存储第一个token的logits
        if i == 0:
            logits_first_token = logits[:, -1, :]  # 只存储当前步最后一个token的logits
        
        # 通过采样策略选择下一个token id
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        
        if next_token_id == tokenizer.eos_token_id: # next_token_id == 2, 表示生成结束
            break

        # 更新input_ids
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        
        # 解码生成的token，添加到文本中
        generated_text += ' ' + tokenizer.decode(next_token_id[0], skip_special_tokens=True)

    gen_text = generated_text[len(prompt):]
    gen_token_id = input_ids[:, input_len:]

    gen_text = gen_text.split()[0]
    if is_multi_token(tokenizer, gen_text):
        print("\t gen_text is multi_token")
        return -1, -1, -1, -1
    
    gen_first_token = gen_token_id[0][0].item()

    probabilities_first_token = torch.softmax(logits_first_token, dim=-1)

    return logits_first_token[0], probabilities_first_token[0], gen_first_token, gen_text

def cal_mDis(str1: str, str2: str) -> int:
    # Check if the two strings are identical
    if str1 == str2:
        return 0  # Return 0 if they are the same
    else:
        return 1

# calc softPred
def cal_norm_of_soft_prediction_diff(O, O_prime):
    """
    计算两个输出O和O'之间的Norm of Soft Prediction Difference
    O和O'为两个模型的输出，形状为(C)，C是类数
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

    count_mDis = 0
    count_qErr = 0
    count_qErr_prime = 0
    count_N_sample = 0


    count_NormSoftPredDiff_logits = 0
    count_NormSoftPredDiff_prob = 0

    with jsonlines.open(file_path) as reader:
        for obj in tqdm(reader):
            task_id = obj.get('task_id').split('/')[1]
            prompt = obj.get('prompt')
            refer = obj.get('canonical_solution')
            print(f"Task ID: {task_id}")
            ground_truth_code = prompt + refer
            masked_results, ground_labels = mask_code_keywords(ground_truth_code)

            
            # 输出每次 mask 掉后的结果
            for idx, (masked_code, ground_label) in enumerate(zip(masked_results, ground_labels), 1):
                print(f"Masked Version {idx}:")

                if is_multi_token(tokenizer1, ground_label):
                    print(f"\t ground_label is multi_token: {ground_label}")
                    continue
                
                # 生成 单个token 的 logits, probabilities, token id, text
                logits1, prob1, token1, gen_text1 = generate_outputs(model1, tokenizer1, masked_code, device1)
                logits2, prob2, token2, gen_text2 = generate_outputs(model2, tokenizer2, masked_code, device2)
                logits2 = logits2.to(logits1.device)
                prob2 = prob2.to(logits1.device)

                # 输出结果
                print(f"\t ground_label: {ground_label}")
                print(f"\t gen_text1: {gen_text1}")
                print(f"\t gen_text2: {gen_text2}")
                print("---------------------------------")
                count_mDis = count_mDis + cal_mDis(gen_text1, gen_text2)
                count_qErr = count_qErr + cal_mDis(gen_text1, ground_label)
                count_qErr_prime = count_qErr_prime + cal_mDis(gen_text2, ground_label)
        
                count_NormSoftPredDiff_logits = count_NormSoftPredDiff_logits + cal_norm_of_soft_prediction_diff(logits1, logits2)
                count_NormSoftPredDiff_prob = count_NormSoftPredDiff_prob + cal_norm_of_soft_prediction_diff(prob1, prob2)

                count_N_sample = count_N_sample + 1

            # if int(task_id) > 4:
            #     break

    m_Dis = count_mDis / count_N_sample

    q_Err = count_qErr / count_N_sample
    m_ErrCorrDis = m_Dis / q_Err
    m_ErrCorrDis2 = count_mDis / count_qErr

    q_Err_prime = count_qErr_prime / count_N_sample
    m_min_Dis = abs(q_Err - q_Err_prime)
    m_max_Dis = min((q_Err + q_Err_prime), 1)
    m_MinMaxNorm_Dis = (m_Dis - m_min_Dis) / (m_max_Dis - m_min_Dis)

    print(f"\t m_Dis: {m_Dis}")
    print(f"\t m_ErrCorrDis: {m_ErrCorrDis}")
    print(f"\t m_ErrCorrDis2: {m_ErrCorrDis2}")
    print(f"\t m_MinMaxNorm_Dis: {m_MinMaxNorm_Dis}")

    m_NormSoftPredDiff_logits = count_NormSoftPredDiff_logits / (2 * count_N_sample)
    m_NormSoftPredDiff_prob = count_NormSoftPredDiff_prob / (2 * count_N_sample)
            
    print(f"\t m_NormSoftPredDiff_logits: {m_NormSoftPredDiff_logits}")
    print(f"\t m_NormSoftPredDiff_prob: {m_NormSoftPredDiff_prob}")

if __name__ == "__main__":

    # 指定GPU设备：
    device_model1 = torch.device("cuda:2")  # 第x块GPU
    device_model2 = torch.device("cuda:3")  # 第y块GPU

    # device_model1 = 'cpu'
    # device_model2 = 'cpu'

    # 设置模型和输入
    model_1 = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b"
    model_2 = "/newdisk/public/wws/model_dir/codellama/CodeLlama-7b-Instruct-hf" # "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b-Python"

    # 打开jsonl文件并遍历
    file_path = '/newdisk/public/wws/humaneval-x-main/data/python/data/humaneval.jsonl'  # Dataset

    main(model_1, model_2, file_path, device_model1, device_model2)
