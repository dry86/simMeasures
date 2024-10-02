from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import time  # 导入 time 模块


# 记录开始时间
start_time = time.time()

model_path = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b"
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda:1")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32
).to(device)

prompt = '''from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> <FILL_ME>:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n
'''

input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
output = model.generate(
    input_ids,
    max_new_tokens=200,
)
output = output[0].to("cpu")

output_decode = tokenizer.decode(output, skip_special_tokens=True)
print(f"output:{output_decode}\n")
filling = tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True)
print(f"filling:{filling}")
print(prompt.replace("<FILL_ME>", filling))


# 记录结束时间
end_time = time.time()

# 计算并打印程序运行时间
elapsed_time = end_time - start_time
print(f"Program runtime: {elapsed_time:.2f} seconds")   