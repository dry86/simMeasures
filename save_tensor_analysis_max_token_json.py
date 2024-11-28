import torch
import pandas as pd
from transformers import AutoTokenizer
from utils import extract_prompts

def main(task, model1_path, model2_path, data_file_path, device1, device2):

    # 加载tokenizer
    tokenizer1 = AutoTokenizer.from_pretrained(model1_path)
    tokenizer2 = AutoTokenizer.from_pretrained(model2_path)

    # 使用 eos_token 作为 pad_token
    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer1.padding_side = "right"   # 使用EOS时,向右填充
    tokenizer2.pad_token = tokenizer2.eos_token
    tokenizer2.padding_side = "right"

    token1 = []
    token2 = []
    # 读取数据文件
    prompts = extract_prompts(task, "python")
    for task_id, prompt in prompts:
        # print(f"Task ID: {task_id}")

        inputs_model1 = tokenizer1(prompt, return_tensors='pt').to(device1)
        token1.append(inputs_model1['input_ids'].cpu().numpy())
        inputs_model2 = tokenizer2(prompt, return_tensors='pt').to(device2)
        token2.append(inputs_model2['input_ids'].cpu().numpy())

    lengths1 = [len(seq[0]) for seq in token1]
    stats = pd.DataFrame(lengths1, columns=['length']).describe(percentiles=[0.9,0.95])
    print(stats)
    # 获取90%的分位数并四舍五入
    percentile_90_1 = round(stats.loc['90%', 'length'])
    print(f"90% percentile for model1: {percentile_90_1}")

    lengths2 = [len(seq[0]) for seq in token2]
    stats = pd.DataFrame(lengths2, columns=['length']).describe(percentiles=[0.9,0.95])
    print(stats)

if __name__ == "__main__":

    """
    how to use:
        修改 'data_file' 要分析的数据集语言, 看此语言数据集在90%情况下token的大小, 然后传给save_tensor.py 中 padding_max_length 
    """
    # 指定GPU设备
    device_model1 = torch.device("cuda:0")
    device_model2 = torch.device("cuda:1")

    # 模型和数据路径
    model_1 = "/newdisk/public/wws/model_dir/StarCoder2/starcoder2-7b"
    model_2 = "/newdisk/public/wws/model_dir/codellama/codeLlama-7b"
    
    data_file = "/newdisk/public/wws/Dataset/CodeSearchNet/dataset/python/test.jsonl"
    task = "codeSummary_CSearchNet"
    # 调用主函数
    main(task, model_1, model_2, data_file, device_model1, device_model2)
    


