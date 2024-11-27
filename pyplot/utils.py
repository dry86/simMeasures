

subtitles = {
    "codeLlama-7b-codeLlama-7b-Python": "codeLlama-7b: Base vs Python", 
    "codeLlama-7b-Python": "codeLlama-7b: Base vs Python", 
    "codeLlama-7b-Instruct-codeLlama-7b-Python": "codeLlama-7b: Instruct vs Python", 
    "codeLlama-7b-Instruct-Python": "codeLlama-7b: Instruct vs Python", 
    "codeLlama-7b-codeLlama-7b-Instruct": "codeLlama-7b: Base vs Instruct", 
    "codeLlama-7b-Instruct": "codeLlama-7b: Base vs Instruct", 
    "dsc-6.7b-base-dsc-6.7b-instruct" : "dsCoder-6.7b: Base vs Instruct", 
    "dsc-6.7b-base-instruct" : "dsCoder-6.7b: Base vs Instruct", 
    "dsc-7b-base-v1.5-dsc-7b-base-instruct-v1.5": "dsCoder-7b: Base vs Instruct", 
    "dsc-7b-base-v1.5-v1.5": "dsCoder-7b: Base vs Instruct", 
    "Qwen2.5-Coder-7B-Qwen2.5-Coder-7B-Instruct": "QwenCoder-7b: Base vs Instruct",
    "Qwen2.5-Coder-7B-Instruct": "QwenCoder-7b: Base vs Instruct",
    "codeLlama-7b-dsc-7b-base-v1.5": "codeLlama-7b vs dsCoder-7b",
    "codeLlama-7b-Qwen2.5-Coder-7B": "codeLlama-7b vs QwenCoder-7b",
    "dsc-7b-base-v1.5-Qwen2.5-Coder-7B": "dsCoder-7b vs QwenCoder-7b"
}

def replace_keys_with_values(input_string):
    # 遍历所有的键和值
    for key, value in subtitles.items():
        # 如果输入字符串中包含该键，则用相应的值替换它
        if key in input_string:
            # input_string = input_string.replace(key, value)
            return value
