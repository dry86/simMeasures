import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# 定义模型加载函数
def load_model_and_tokenizer(model_path: str, device: torch.device):
    """
    加载预训练的模型和分词器，并将模型加载到指定设备上。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32
    ).to(device)
    return model, tokenizer

# 定义生成文本函数
def generate_text(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int = 200):
    """
    基于给定的 prompt 生成新的文本。
    """
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens)
    output = output[0].to("cpu")  # 将结果转移到 CPU
    generated_text = tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text

# 主函数：执行代码并输出结果
def main():
    # 模型路径和设备设置
    model_path = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    
    # 提示输入 (Prompt)
    prompt = '''<FILL_ME> typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n'''

    # 生成填充内容
    filling = generate_text(model, tokenizer, prompt, device)
    
    # 输出结果
    print(f"filling: {filling}")
    print(prompt.replace("<FILL_ME>", filling))

if __name__ == "__main__":

    start_time = time.time()

    main()

    # 记录结束时间
    end_time = time.time()

    # 计算并打印程序运行时间
    elapsed_time = end_time - start_time
    print(f"Program runtime: {elapsed_time:.2f} seconds")


