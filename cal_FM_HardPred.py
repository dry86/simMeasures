import torch
from getHiddenStates import load_model, get_hidden_states
import numpy as np
import jsonlines
import re

def mask_code_keywords(code: str) -> list:
    # 匹配 """ 注释的正则表达式
    comment_pattern = r'""".*?"""'

    # 找到所有 """ 注释内容
    comments = [match.span() for match in re.finditer(comment_pattern, code, re.DOTALL)]

    # 列表存储多次 mask 的结果
    masked_versions = []

    # 找到所有非注释区域的代码关键字和符号
    # 使用正则表达式捕获关键字和操作符，比如 return、标识符和操作符
    keyword_pattern = r'->|>=|<=|==|!=|\d+\.\d+|\d+|\b\w+\b|[%=+*/-]'  # 捕获字母、数字、和常见操作符

    # 提取所有关键字的位置
    mask_positions = [(match.group(), match.span()) for match in re.finditer(keyword_pattern, code)]

    # 过滤掉注释部分的关键字
    non_comment_positions = [(word, span) for word, span in mask_positions
                             if not any(start <= span[0] < end for start, end in comments)]

    # 对每个关键字进行 mask，生成不同的版本
    for word, (start, end) in non_comment_positions:
        # 生成代码副本并进行 mask
        new_code = list(code)  # 把代码转为字符列表，便于替换
        new_code[start:end] = '<MASK>'  # 替换当前位置的关键字为 <MASK>
        masked_versions.append("".join(new_code))

        # # 截断<MASK>之后的部分
        # truncated_code = "".join(new_code[:end])

        # # 将截断后的版本加入到结果列表
        # masked_versions.append(truncated_code)

        

    return masked_versions

# 指定GPU设备：
device_model1 = torch.device("cuda:0")  # 第x块GPU
device_model2 = torch.device("cuda:3")  # 第y块GPU

# 设置模型和输入
model_7b        = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b"
model_7b_Python = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b-Python"

model1, tokenizer1 = load_model(model_7b, device_model1)
model2, tokenizer2 = load_model(model_7b_Python, device_model2)

model1.eval()
model2.eval()

# 打开jsonl文件并遍历
file_path = '/newdisk/public/wws/humaneval-x-main/data/python/data/humaneval.jsonl'  # Dataset

with jsonlines.open(file_path) as reader:
    for obj in reader:
        task_id = obj.get('task_id')
        prompt = obj.get('prompt')
        refer = obj.get('canonical_solution')
        print(f"Task ID: {task_id}, Prompt: \n{prompt}")
        ground_truth_code = prompt + refer
        masked_results = mask_code_keywords(ground_truth_code)
        prompt = "\n# Continue the code after <MASK>:"
        masked_results = [result + prompt for result in masked_results]
        # 输出每次 mask 掉后的结果
        for idx, masked_code in enumerate(masked_results, 1):
            print(f"Masked Version {idx}:\n{masked_code}\n")

            # 将 "<mask>" 替换为模型能够识别的特殊 token，比如 "<unk>" 或者 "<mask>"，具体取决于模型支持的 token
            # masked_input = masked_code.replace("<mask>", '<MASK>')  # 使用 <unk> 作为占位符

            # inputs1 = tokenizer1(masked_code, return_tensors='pt').to(device_model1)
            # with torch.no_grad():
            #     output_model1 = model1(**inputs1)
            # mask_logits = output_model1.logits
            # predicted_token_id = torch.argmax(mask_logits, dim=-1).item()
            # predicted_token = tokenizer1.decode(predicted_token_id)

            inputs1 = tokenizer1(masked_code, return_tensors='pt').to(device_model1)
            # 找到 <mask> 的位置
            mask_token_index1 = torch.where(inputs1.input_ids == tokenizer1.convert_tokens_to_ids("<MASK>"))[1]
            with torch.no_grad():
                output_model1 = model1(**inputs1)
            # # 获取 mask 位置的 logits
            # mask_logits1 = output_model1.logits[0, mask_token_index1, :]
            # # 获取 mask 的概率矩阵
            # probabilities_model1 = torch.softmax(mask_logits1, dim=-1)
            # # 获取预测的 token id
            # predicted_token_id1 = torch.argmax(mask_logits1, dim=-1).item()
            # # 获取该 token id 对应的 token
            # predicted_token1 = tokenizer1.decode(predicted_token_id1)
            # 获取生成的 token ids（假设是自回归模型，如 CodeLlama 生成了一段序列）
            # 通常 output_model1 的 logits 可以通过 argmax 获取最可能的 token ids
            generated_token_ids1 = torch.argmax(output_model1.logits, dim=-1)
            # 直接使用 tokenizer2.decode() 将生成的 token ids 转换为文本
            generated_text1 = tokenizer1.decode(generated_token_ids1[0], skip_special_tokens=True)
            # 打印生成的文本
            print("Generated text1:", generated_text1)


            # 第二个模型进行相同的处理:
            inputs2 = tokenizer2(masked_code, return_tensors='pt').to(device_model2)
            # with torch.no_grad():
            #     output_model2 = model2(**inputs2)
            # # 找到 <mask> 的位置
            # mask_token_index2 = torch.where(inputs2.input_ids == tokenizer2.convert_tokens_to_ids("<MASK>"))[1]   

            # # 获取 mask 位置的 logits
            # mask_logits2 = output_model2.logits[0, mask_token_index2, :]
            # # 获取 mask 的概率矩阵
            # probabilities_model2 = torch.softmax(mask_logits2, dim=-1)
            # # 获取预测的 token id
            # predicted_token_id2 = torch.argmax(mask_logits2, dim=-1).item()
            # # 获取该 token id 对应的 token
            # predicted_token2 = tokenizer2.decode(predicted_token_id2)
            # # 获取生成的 token ids（假设是自回归模型，如 CodeLlama 生成了一段序列）
            # # 通常 output_model2 的 logits 可以通过 argmax 获取最可能的 token ids
            # generated_token_ids2 = torch.argmax(output_model2.logits, dim=-1)
            # # 直接使用 tokenizer2.decode() 将生成的 token ids 转换为文本
            # generated_text2 = tokenizer2.decode(generated_token_ids2[0], skip_special_tokens=True)
            # # 打印生成的文本
            # print("Generated text2:", generated_text2)


            # 假设使用 generate() 生成了一个序列
            with torch.no_grad():
                generated_token_ids21 = model2.generate(**inputs2, max_length=512)

            # 直接解码生成的 token ids
            generated_text21 = tokenizer2.decode(generated_token_ids21[0], skip_special_tokens=True)

            # 打印生成的文本
            print("Generated text21:", generated_text21)


            





