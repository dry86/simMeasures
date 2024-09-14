from transformers import AutoTokenizer
import transformers
import torch
import time  # 导入 time 模块


# 记录开始时间
start_time = time.time()

model_path = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    torch_dtype=torch.float32,
    device="cuda:3"
    # device_map="auto",
    # device_map={"": "cuda:0", "encoder": "cuda:0", "decoder": "cuda:1"}  # 指定 GPU
)

sequences = pipeline(
    'def fibonacci(',
    do_sample=True,
    temperature=0.2,
    top_p=0.9,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=512,
    truncation=True,
)

# 记录结束时间
end_time = time.time()

for seq in sequences:
    print(f"Result: {seq['generated_text']}")

# 计算并打印程序运行时间
elapsed_time = end_time - start_time
print(f"Program runtime: {elapsed_time:.2f} seconds")

