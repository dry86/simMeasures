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

    data_file_path = f"/newdisk/public/wws/Dataset/humaneval-x-main/data/{language.lower()}/data/humaneval.jsonl"

    pt_dir = f"/newdisk/public/wws/simMeasures/pt_file/test/{lang}/"
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
                # attention mask debug
                # attention_mask = inputs_model1["attention_mask"]
                # torch.set_printoptions(threshold=torch.nan)  # 设置为无限制，显示所有元素
                # print(attention_mask.cpu().numpy())
                # attention mask debug end
                
                # 获取隐藏层输出
                outputs, hidden_states_model1 = tokens_get_hidden_states(model1, inputs_model1, device1)

                # 获取logits
                logits = outputs.logits  # shape = (batch_size, seq_len, vocab_size)
                
                # 获取每个位置上最可能的token的id
                predicted_token_ids = torch.argmax(logits, dim=-1)  # shape = (batch_size, seq_len)
                
                # 使用tokenizer将token id转换为文本
                predicted_texts = [tokenizer1.decode(ids, skip_special_tokens=True) for ids in predicted_token_ids]

                # 打印转换后的文本
                for text in predicted_texts:
                    print(text)
                
                # 保存 hidden_states 到文件
                # torch.save(hidden_states_model1, f"{save_dir}{model_idx}_batch_{task_number}.pt")

                # 清空prompts，准备下一个batch
                prompts = []
                break



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
    device_model = torch.device("cuda:3")

    # 模型和数据路径
    model_path = "/newdisk/public/wws/model_dir/codellama/codeLlama-7b-Instruct"
    model_idx = "codeLlama-7b-Instruct"
    
    # 设置的数据集语言
    lang = "Python"
    
    # 调用主函数
    main(model_path, model_idx, lang, padding_max_length[lang], device_model)
    
