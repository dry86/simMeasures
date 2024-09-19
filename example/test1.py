import evaluate

# print(evaluate.list_evaluation_modules())

# 加载 BLEU 指标
bleu = evaluate.load("bleu")
# 如果卡在load函数上, 将下面一行代码在bash上运行,再运行此.py
# export HF_ENDPOINT=https://hf-mirror.com

# 参考文本和生成文本
references = [["This is a small cat.", "That is a cat."]]  # 可以包含多个参考句子
predictions = ["This is a tiny cat."]

# 计算 BLEU 分数
results = bleu.compute(predictions=predictions, references=references)

print(results['bleu'])