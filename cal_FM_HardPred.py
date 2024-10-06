import torch
from getHiddenStates import load_model, get_hidden_states
from transformers import AutoTokenizer, AutoModelForCausalLM
import jsonlines
import re

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

# 定义生成文本函数
def generate_text(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int = 10):
    """
    基于给定的 prompt 生成新的文本。
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    output = model.generate(input_ids, 
                            attention_mask = attention_mask,
                            max_new_tokens = max_new_tokens, do_sample=True, top_p=0.9, temperature=0.1) #, do_sample=True, top_p=0.9, temperature=0.1, num_return_sequences=1, repetition_penalty=0.9, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id 
    output = output[0].to("cpu")  # 将结果转移到 CPU
    generated_text = tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text

def cal_mDis(str1: str, str2: str) -> int:
    # Check if the two strings are identical
    if str1 == str2:
        return 0  # Return 0 if they are the same
    else:
        return 1


def main(model_1, model_2, file_path, device1, device2):

    model1, tokenizer1 = load_model_and_tokenizer(model_1, device1)
    model2, tokenizer2 = load_model_and_tokenizer(model_2, device2)

    model1.eval()
    model2.eval()

    count_mDis = 0
    count_qErr = 0
    count_qErr_prime = 0
    count_N_sample = 0

    with jsonlines.open(file_path) as reader:
        for obj in reader:
            task_id = obj.get('task_id').split('/')[1]
            prompt = obj.get('prompt')
            refer = obj.get('canonical_solution')
            # print(f"Task ID: {task_id}, Prompt: \n{prompt}")
            ground_truth_code = prompt + refer
            masked_results, ground_labels = mask_code_keywords(ground_truth_code)

            
            # 输出每次 mask 掉后的结果
            for idx, (masked_code, ground_label) in enumerate(zip(masked_results, ground_labels), 1):
                print(f"Masked Version {idx}:")
                # print(f"{masked_code}\n")

                # 生成填充内容
                filling1 = generate_text(model1, tokenizer1, masked_code, device1)
                filling2 = generate_text(model2, tokenizer2, masked_code, device2)
                # 取生成的第一个内容
                token1 = re.search(keyword_pattern, filling1).group()
                token2 = re.search(keyword_pattern, filling2).group()

                # 输出结果
                print(f"\t ground_label: {ground_label}")
                print(f"\t filling_1: {token1}")
                print(f"\t filling_2: {token2}")
                print("---------------------------------")
                count_mDis = count_mDis + cal_mDis(token1, token2)
                count_qErr = count_qErr + cal_mDis(token1, ground_label)
                count_qErr_prime = count_qErr_prime + cal_mDis(token2, ground_label)
        
            count_N_sample = count_N_sample + len(masked_results)

            if int(task_id) > 5:
                break

    m_Dis = count_mDis / count_N_sample

    q_Err = count_qErr / count_N_sample
    m_ErrCorrDis = m_Dis / q_Err
    m_ErrCorrDis2 = count_mDis / count_qErr

    q_Err_prime = count_qErr_prime / count_N_sample
    m_min_Dis = abs(q_Err - q_Err_prime)
    m_max_Dis = min((q_Err + q_Err_prime), 1)
    m_MinMaxNorm_Dis = (m_Dis - m_min_Dis) / (m_max_Dis - m_min_Dis)

    print(m_Dis)
    print(m_ErrCorrDis)

            

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

            





