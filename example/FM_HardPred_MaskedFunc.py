import torch
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


# 打开jsonl文件并遍历
file_path = '/newdisk/public/wws/humaneval-x-main/data/python/data/humaneval.jsonl'  # Dataset

with jsonlines.open(file_path) as reader:
    for obj in reader:
        task_id = obj.get('task_id')
        prompt = obj.get('prompt')
        refer = obj.get('canonical_solution')
        print(f"Task ID: {task_id}\n")
        ground_truth_code = prompt + refer
        masked_results = mask_code_keywords(ground_truth_code)

        # 输出每次 mask 掉后的结果
        for idx, masked_code in enumerate(masked_results, 1):
            print(f"Masked Version {idx}:\n{masked_code}\n")

        break

