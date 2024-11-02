import torch
from getHiddenStates import load_model, tokens_get_hidden_states
import numpy as np
import jsonlines
from tqdm import tqdm
import pandas as pd
# 设置打印选项来显示所有元素
# torch.set_printoptions(threshold=torch.inf)


def pad_to_max_length(tensor_list, tokenizer):
    # 获取所有张量的最大长度
    max_length = max(tensor.shape[1] for tensor in tensor_list)
    
    # 对每个张量进行填充，使它们具有相同的长度
    padded_tensors = [torch.nn.functional.pad(tensor, (0, 0, 0, max_length - tensor.shape[1])) for tensor in tensor_list]   # , value=tokenizer.eos_token_id
    
    return torch.stack(padded_tensors)

def main(model1_path, model2_path, data_file_path, device1, device2):

    # 加载模型和tokenizer
    model1, tokenizer1 = load_model(model1_path, device1)
    model2, tokenizer2 = load_model(model2_path, device2)
    
    # 使用 eos_token 作为 pad_token
    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer1.padding_side = "right"   # 使用EOS时,向右填充
    tokenizer2.pad_token = tokenizer2.eos_token
    tokenizer2.padding_side = "right"

    token1 = []
    token2 = []
    # 读取数据文件
    with jsonlines.open(data_file_path) as reader:
        for obj in reader:
            # task_id = obj.get('id')
            # prompt = obj.get('input')
            # print(f"Task ID: {task_id}")
            task_id = obj.get('repo')
            prompt = "please describe the functionality of the method: " + obj.get('code')
            print(f"Task ID: {task_id}")


            inputs_model1 = tokenizer1(prompt, return_tensors='pt').to(device1)
            token1.append(inputs_model1['input_ids'].cpu().numpy())
            inputs_model2 = tokenizer2(prompt, return_tensors='pt').to(device2)
            token2.append(inputs_model2['input_ids'].cpu().numpy())

    lengths1 = [len(seq[0]) for seq in token1]
    stats = pd.DataFrame(lengths1, columns=['length']).describe(percentiles=[0.9,0.95])
    print(stats)
    lengths2 = [len(seq[0]) for seq in token2]
    stats = pd.DataFrame(lengths2, columns=['length']).describe(percentiles=[0.9,0.95])
    print(stats)

if __name__ == "__main__":

    """
    how to use:
        修改 'data_file' 要分析的数据集语言, 看此语言数据集在90%情况下token的大小, 然后传给save_tensor.py 中 padding_max_length 
    """
    # 指定GPU设备
    device_model1 = torch.device("cuda:2")
    device_model2 = torch.device("cuda:3")

    # 模型和数据路径
    model_1 = "/newdisk/public/wws/model_dir/Qwen2.5-Coder/Qwen2.5-Coder-7B"
    model_2 = "/newdisk/public/wws/model_dir/Qwen2.5-Coder/Qwen2.5-Coder-7B-Instruct"
    
    data_file = "/newdisk/public/wws/Dataset/CodeSearchNet/dataset/java/test.jsonl"

    # 调用主函数
    main(model_1, model_2, data_file, device_model1, device_model2)
    


