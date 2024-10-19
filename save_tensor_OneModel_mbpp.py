import torch
from getHiddenStates import load_model, get_last_layer_hidden_states
import jsonlines
import os

def main(model1_path, model_idx, padding_len, device1, batch_size=20):

    # 加载模型和tokenizer
    model1, tokenizer1 = load_model(model1_path, device1)

    # 使用 eos_token 作为 pad_token
    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer1.padding_side = "right"   # 使用EOS时,向右填充

    prompts = [] 

    data_file_path = f"/newdisk/public/wws/Dataset/mbpp/mbpp.jsonl"

    pt_dir = f"/newdisk/public/wws/simMeasures/pt_file/mbpp/"
    save_dir = pt_dir + model_idx + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 读取数据文件
    with jsonlines.open(data_file_path) as reader:
        for obj in reader:
            task_id = obj.get('task_id')
            prompt = obj.get('text')
            print(f"Task ID: {task_id}")
            prompts.append(prompt)
            
            if len(prompts) == batch_size or task_id == 974:  # 分批加载

                # 获取所有 prompts 的输入张量，并进行填充
                inputs_model1 = tokenizer1(prompts, 
                                           return_tensors='pt',
                                           padding='max_length', 
                                           max_length=padding_len, 
                                           truncation=True 
                                           ).to(device1)

                # 获取隐藏层输出
                outputs, last_layer_hidden_states = get_last_layer_hidden_states(model1, inputs_model1)

                # 保存 hidden_states 到文件
                # torch.save(last_layer_hidden_states, f"{save_dir}{model_idx}_batch_{task_number}.pt")

                # 清空prompts，准备下一个batch
                prompts = []
                # break



if __name__ == "__main__":
    """
    how to use:
        修改以下参数↓↓↓
    """

    padding_max_length = 25 # mbpp text 90%: 22, 95%: 25

    # 指定GPU设备
    device_model = torch.device("cuda:3")

    # 模型和数据路径
    model_path = "/newdisk/public/wws/model_dir/codellama/codeLlama-7b-Instruct"
    model_idx = "codeLlama-7b-Instruct"
    
    # 调用主函数
    main(model_path, model_idx, padding_max_length, device_model)
    
