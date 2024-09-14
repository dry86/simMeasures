from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import time  # 导入 time 模块


# 记录开始时间
start_time = time.time()

model_path = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b"
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32
).to("cuda")

prompt = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''

input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
output = model.generate(
    input_ids,
    max_new_tokens=200,
)
output = output[0].to("cpu")

filling = tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True)
print(prompt.replace("<FILL_ME>", filling))


# 记录结束时间
end_time = time.time()

# 计算并打印程序运行时间
elapsed_time = end_time - start_time
print(f"Program runtime: {elapsed_time:.2f} seconds")