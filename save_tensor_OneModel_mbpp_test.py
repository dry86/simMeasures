import torch
from getHiddenStates import load_model, get_last_layer_hidden_states
import jsonlines
import os

def main(model1_path, model_idx, padding_len, device1, batch_size=1):

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
                                           ).to(device1)

                # 获取隐藏层输出
                outputs, last_layer_hidden_states = get_last_layer_hidden_states(model1, inputs_model1)

                # 获取logits
                logits = outputs.logits  # shape = (batch_size, seq_len, vocab_size)
                
                # 获取每个位置上最可能的token的id
                predicted_token_ids = torch.argmax(logits, dim=-1)  # shape = (batch_size, seq_len)
                
                # 使用tokenizer将token id转换为文本
                predicted_texts = [tokenizer1.decode(ids, skip_special_tokens=True) for ids in predicted_token_ids]

                # 打印转换后的文本
                for text in predicted_texts:
                    print(text)


                generated_outputs = model1.generate(
                    **inputs_model1,
                    max_length=512,              # 设置生成文本的最大长度
                    num_return_sequences=1,                # 设置生成的序列数量
                    do_sample=True,                        # 使用采样生成文本（非贪婪算法）
                    top_p=0.9,                             # 设置Top-p采样
                    temperature=1.0                        # 温度参数，控制生成的多样性
                )

                # 解码生成的 token 为可读的文本
                generated_texts = [tokenizer1.decode(output, skip_special_tokens=True) for output in generated_outputs]

                # 打印生成的文本
                for text in generated_texts:
                    print(text)     

                print("--------------------")


                # 获取初始输入序列的长度
                input_seq_len = inputs_model1['input_ids'].shape[1]
                # 定义生成的最大序列长度
                max_generate_length = 512  # 假设我们希望生成最多50个token
                generated_sequences = []
                # 将初始输入序列复制为生成序列的起点
                generated_sequences = inputs_model1['input_ids']

                for step in range(max_generate_length):
                    # 获取隐藏层输出
                    outputs, last_layer_hidden_states = get_last_layer_hidden_states(model1, inputs_model1)
                    
                    # 获取 logits 并找到每个输入序列最后一个位置的下一个 token 的预测 id
                    logits = outputs.logits  # shape = (batch_size, seq_len, vocab_size)
                    next_token_id = torch.argmax(logits[:, -1], dim=-1)  # shape = (batch_size,)

                    next_token_texts = [tokenizer1.decode(token_id, skip_special_tokens=True) for token_id in next_token_id]
                    print(next_token_texts)
                    
                    # 将 next_token_id 拼接到当前的输入序列
                    generated_sequences = torch.cat([generated_sequences, next_token_id.unsqueeze(-1)], dim=-1)
                    
                    # 使用拼接后的序列作为新的输入
                    inputs_model1 = {
                        'input_ids': generated_sequences,
                        'attention_mask': torch.ones_like(generated_sequences).to(device1)  # 更新 attention mask
                    }

                    # 可选：检测是否生成了结束标记
                    # 如果你有指定的结束标记（如 <eos>），可以在此终止循环
                    if (next_token_id == tokenizer1.eos_token_id).all():
                        break

                # 最终生成的序列
                generated_sequences = generated_sequences.cpu().numpy()

                # 解码生成的 token 序列为文本
                generated_texts = [tokenizer1.decode(seq, skip_special_tokens=True) for seq in generated_sequences]

                # 输出生成的文本
                for text in generated_texts:
                    print(text)








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
    
