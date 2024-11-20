import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model_path = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b"
# model_path = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b-Python"
# model_path = "/newdisk/public/wws/codeLlama/codellama/CodeLlama-7b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             torch_dtype=torch.float32)

# 将模型转移到指定的 GPU 设备 cuda:3
device = torch.device("cuda:1")
model.to(device)

# 模型设置为评估模式
model.eval()

# 准备输入
input_text = "def fibonacci("
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# 获取模型输出
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

# 提取 logits 和 hidden states
logits = outputs.logits
hidden_states = outputs.hidden_states

# 打印输出维度
print("Logits shape:", logits.shape)
print("Logits for the last token:", logits[:, -1, :])

print("Number of layers:", len(hidden_states))
print("Shape of hidden states for the last layer:", hidden_states[-1].shape)