import torch
from getHiddenStates import load_model, tokens_get_hidden_states
import numpy as np
import jsonlines
from example import cca_core
from tqdm import tqdm
# 设置打印选项来显示所有元素
# torch.set_printoptions(threshold=torch.inf)


def pad_to_max_length(tensor_list, tokenizer):
    # 获取所有张量的最大长度
    max_length = max(tensor.shape[1] for tensor in tensor_list)
    
    # 对每个张量进行填充，使它们具有相同的长度
    padded_tensors = [torch.nn.functional.pad(tensor, (0, 0, 0, max_length - tensor.shape[1])) for tensor in tensor_list]   # , value=tokenizer.eos_token_id
    
    return torch.stack(padded_tensors)

def main(model1_path, model2_path, data_file_path, device1, device2, batch_size=20):

    # 加载模型和tokenizer
    model1, tokenizer1 = load_model(model1_path, device1)
    model2, tokenizer2 = load_model(model2_path, device2)
    
    # 使用 eos_token 作为 pad_token
    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer2.pad_token = tokenizer2.eos_token
    tokenizer1.padding_side = "right"   # 使用EOS时,向右填充
    tokenizer2.padding_side = "right"

    prompts = []
    padding_max_length = 0
    with jsonlines.open(data_file_path) as reader:
        for obj in reader:
            prompt = obj.get('prompt')
            prompt_length = tokenizer1(prompt, return_tensors='pt').input_ids.shape[1]
            padding_max_length = max(padding_max_length, prompt_length)

    # 读取数据文件
    with jsonlines.open(data_file_path) as reader:
        for obj in reader:
            task_id = obj.get('task_id')
            task_number = int(task_id.split('/')[-1])
            prompt = obj.get('prompt')
            print(f"Task ID: {task_id}")
            # print(f"Prompt: \n{prompt}")
            prompts.append(prompt)
            
            if len(prompts) == batch_size or task_number == 163:  # 分批加载

                # 获取所有 prompts 的输入张量，并进行填充
                inputs_model1 = tokenizer1(prompts, return_tensors='pt', padding='max_length', max_length=padding_max_length).to(device1)
                inputs_model2 = tokenizer2(prompts, return_tensors='pt', padding='max_length', max_length=padding_max_length).to(device2)

                # 获取隐藏层输出
                hidden_states_model1 = tokens_get_hidden_states(model1, inputs_model1, device1)
                hidden_states_model2 = tokens_get_hidden_states(model2, inputs_model2, device2)

                # 保存 hidden_states 到文件
                torch.save(hidden_states_model1, f"./pt_file/{task_id.split('/')[0]}_hsm1_batch_{task_number}.pt")
                torch.save(hidden_states_model2, f"./pt_file/{task_id.split('/')[0]}_hsm2_batch_{task_number}.pt")

                # 清空prompts，准备下一个batch
                prompts = []

 

if __name__ == "__main__":
    # 指定GPU设备
    device_model1 = torch.device("cuda:0")
    device_model2 = torch.device("cuda:1")

    # 模型和数据路径
    model_7b = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b"
    model_7b_Python = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b-Python"
    
    data_file = "/newdisk/public/wws/humaneval-x-main/data/js/data/humaneval.jsonl"

    # 调用主函数
    main(model_7b, model_7b_Python, data_file, device_model1, device_model2)
    

