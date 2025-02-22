import torch
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# 加载大规模模型
model_name_1 = "/newdisk/public/wws/00-Model-AIGC/Llama-2-7b-hf"  # 假设模型支持
model_name_2 = "/newdisk/public/wws/00-Model-AIGC/deepseek-llm-7b-base"  # 假设模型支持

# 问题1：应该为每个模型单独加载对应的tokenizer
tokenizer1 = AutoTokenizer.from_pretrained(model_name_1)
tokenizer1.pad_token = tokenizer1.eos_token  # 使用eos_token作为pad_token

tokenizer2 = AutoTokenizer.from_pretrained(model_name_2)
tokenizer2.pad_token = tokenizer2.eos_token  # 使用eos_token作为pad_token

# 问题2：未指定模型加载位置（CPU/GPU）
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model1 = AutoModelForSequenceClassification.from_pretrained(model_name_1).to(device1)
model2 = AutoModelForSequenceClassification.from_pretrained(model_name_2).to(device2)  # 添加设备位置

# 假设数据集是GLUE任务中的MRPC数据
texts = ["The quick brown fox jumped over the lazy dog.", "I love programming with Python."]


# 获取两个模型的输出
# 问题3：两个模型应该使用各自的tokenizer处理输入
with torch.no_grad():
    inputs1 = tokenizer1(texts, padding=True, truncation=True, return_tensors="pt").to(device1)  # 使用tokenizer1
    inputs2 = tokenizer2(texts, padding=True, truncation=True, return_tensors="pt").to(device2)  # 使用tokenizer2
    
    logits_1 = model1(**inputs1).logits
    logits_2 = model2(**inputs2).logits  # 使用对应的输入

# 计算CKA相似度
def cka(X, Y):
    # PCA降维至相同的维度
    pca = PCA(n_components=100)
    X_pca = pca.fit_transform(X)
    Y_pca = pca.fit_transform(Y)
    # 计算内积
    return np.dot(X_pca.T, Y_pca) / (np.linalg.norm(X_pca) * np.linalg.norm(Y_pca))

# 将logits转换为numpy
logits_1 = logits_1.numpy()
logits_2 = logits_2.numpy()

# 计算CKA相似度
similarity = cka(logits_1, logits_2)
print(f"CKA Similarity between Llama3-7b and Deepseek-LLM-7b: {similarity}")