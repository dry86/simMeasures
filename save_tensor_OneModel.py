import torch
from getHiddenStates import load_model, tokens_get_hidden_states
import jsonlines
import os

def main(model1_path, model_idx, language, padding_len, device1, batch_size=20):

    # 加载模型和tokenizer
    model1, tokenizer1 = load_model(model1_path, device1)

    # 使用 eos_token 作为 pad_token
    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer1.padding_side = "right"   # 使用EOS时,向右填充

    prompts = [] 

    data_file_path = f"/newdisk/public/wws/humaneval-x-main/data/{language.lower()}/data/humaneval.jsonl"

    pt_dir = f"/newdisk/public/wws/simMeasures/pt_file/{lang}/"
    save_dir = pt_dir + model_idx + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 读取数据文件
    with jsonlines.open(data_file_path) as reader:
        for obj in reader:
            task_id = obj.get('task_id')
            task_number = int(task_id.split('/')[-1])
            prompt = obj.get('prompt')
            print(f"Task ID: {task_id}")
            prompts.append(prompt)
            
            if len(prompts) == batch_size or task_number == 163:  # 分批加载

                # 获取所有 prompts 的输入张量，并进行填充
                inputs_model1 = tokenizer1(prompts, 
                                           return_tensors='pt', 
                                           padding='max_length', 
                                           max_length=padding_len, 
                                           truncation=True
                                           ).to(device1)

                # 获取隐藏层输出
                hidden_states_model1 = tokens_get_hidden_states(model1, inputs_model1, device1)

                # 保存 hidden_states 到文件
                torch.save(hidden_states_model1, f"{save_dir}{model_idx}_batch_{task_number}.pt")

                # 清空prompts，准备下一个batch
                prompts = []



if __name__ == "__main__":
    """
    how to use:
        修改以下四个参数↓↓↓
    """

    padding_max_length = {  # python 90%: 262, cpp 90%: 275, java 90%: 292, javascript 90%: 259, go 90%: 168
    "Python": 262,
    "CPP": 275,
    "Java": 292,
    "JavaScript": 259,
    "GO": 168
}   


    # 指定GPU设备
    device_model = torch.device("cuda:0")

    # 模型和数据路径
    model_path = "/newdisk/public/wws/model_dir/codellama/codeLlama-13b"
    model_idx = "codeLlama-13b"
    
    # 设置的数据集语言
    lang = "Python"
    
    # 调用主函数
    main(model_path, model_idx, lang, padding_max_length[lang], device_model)
    
