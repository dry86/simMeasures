import torch
from getHiddenStates import load_model, tokens_get_hidden_states
import jsonlines
import os
import time
import json5

def main(model1_path, model_idx, padding_len, lang, device1, batch_size=1):

    # 加载模型和tokenizer
    model, tokenizer1 = load_model(model1_path, device1)

    # 使用 eos_token 作为 pad_token
    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer1.padding_side = "right"   # 使用EOS时,向右填充

    data_file_path = f"/newdisk/public/wws/Dataset/mbpp/mbpp.jsonl" # M

    save_dir = model_path + "/pt_file" + "/textGen_MBPP/" +  lang + "/" # M
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 存储每次获取的 last_token_hidden_states
    accumulated_hidden_states = []
    batch_counter = 0

    batch_idx = 1
    print(f"\tbatch process start!")
    # 读取数据文件
    with jsonlines.open(data_file_path) as reader:
        for obj in reader:
            task_id = obj.get('task_id')
            prompt = obj.get('text')
            # print(f"Task ID: {task_id}")

            inputs = tokenizer1(prompt, 
                                return_tensors='pt',
                                padding='max_length', 
                                max_length=padding_len, 
                                truncation=True
                                ).to(device1)

            # 获取隐藏层输出
            hidden_states = tokens_get_hidden_states(model, inputs)

            # 获取每一层最后一个有效(非padding)token
            last_non_padding_index = inputs['attention_mask'].sum(dim=1) - 1
            last_token_hidden_states = [layer_output[torch.arange(batch_size), last_non_padding_index, :].squeeze(0).cpu() for layer_output in hidden_states]

            # 将每层的 last_token_hidden_states 添加到累积列表
            accumulated_hidden_states.append(last_token_hidden_states)
            batch_counter += batch_size

            # 当累积数量达到1000时，保存hidden states并清空累积
            if batch_counter >= 1000:
                # 将累积的 hidden states 转换为张量 (1000, num_layers, hidden_size)
                concatenated_hidden_states = torch.stack([torch.stack(states) for states in accumulated_hidden_states])

                # 进行形状变换，使得保留33层, 每层形状为(1000, 4096)
                concatenated_hidden_states = concatenated_hidden_states.permute(1, 0, 2)  # 变换为 (num_layers, 1000, 4096)

                # 保存拼接的 hidden states 到文件
                torch.save(concatenated_hidden_states, f"{save_dir}{model_idx}_batch_{batch_idx}.pt")
                print(f"\tbatch_{batch_idx} saved!")
                batch_idx = batch_idx + 1

                # 清空累积列表和计数器
                accumulated_hidden_states.clear()
                batch_counter = 0
            
            del hidden_states, last_token_hidden_states
            torch.cuda.empty_cache()
            


    # 如果循环结束后仍有未保存的hidden states
    if accumulated_hidden_states:
        concatenated_hidden_states = torch.stack([torch.stack(states) for states in accumulated_hidden_states])
        concatenated_hidden_states = concatenated_hidden_states.permute(1, 0, 2)  # 变换为 (33, 1000, 4096)
        torch.save(concatenated_hidden_states, f"{save_dir}{model_idx}_batch_{batch_idx}.pt")
        print(f"\tbatch_{batch_idx} saved!")




if __name__ == "__main__":
    """
    how to use:
        修改以下参数↓↓↓
    """
    # 记录开始时间
    start_time = time.time()  


    # 指定GPU设备
    device_model = torch.device("cuda:3")


    # 参数设置
    configs = json5.load(open('/newdisk/public/wws/simMeasures/config/config-save-textGen-MBPP.json5')) # M

    for config in configs:
        model_idx = config.get('model_idx')
        lang = config.get('lang')
        print(f"task list: {model_idx}, {lang}")
    print("-"*50)

    for config in configs:
        model_path = config.get('model_path')
        model_idx = config.get('model_idx')
        
        padding_max_length = config.get('padding_max_length')
        lang = config.get('lang')
        # 调用主函数
        print(f"Current work: {model_idx}")
        main(model_path, model_idx, padding_max_length, lang, device_model)
        
        # 记录结束时间
        end_time = time.time()
        # 计算并打印程序运行时间
        elapsed_time = (end_time - start_time) / 60
        print(f"Program runtime: {elapsed_time:.2f} mins")  