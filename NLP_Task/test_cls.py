from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# 从本地文件夹加载SST-2数据集
dataset = load_dataset(
    "parquet", 
    data_files={
        "validation": "/root/projects/00-Model/00-Dataset/sst2/data/validation-00000-of-00001.parquet",
        # "test": "/root/projects/00-Model/00-Dataset/sst2/data/test-00000-of-00001.parquet"
    }
)

# 加载预训练模型和分词器
model_name = "/root/projects/00-Model-Code/Qwen2.5-Coder-0.5B-Instruct"  # 可替换为其他大模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

# 对文本进行预处理
def tokenize_function(examples):
    result = tokenizer(examples["sentence"], padding="max_length", truncation=True)
    # 保证label字段被正确带入，并转为int
    result["labels"] = examples["label"]
    return result


tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence", "label"]  # 只保留tokenizer输出和labels
)

# 准备评估数据
eval_dataset = tokenized_datasets["validation"]

# 创建数据加载器
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=1, collate_fn=data_collator
)

# 模型评估
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

predictions = []
true_labels = []


for batch in tqdm(eval_dataloader, desc="评估中"):
    # 只保留模型需要的输入
    inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"] and v is not None}
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1)
    predictions.extend(preds.cpu().numpy())
    # label 也要转到 cpu
    true_labels.extend(batch["labels"].cpu().numpy())

# 计算评估指标
accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy:.4f}")

# 打印详细分类报告
print("Classification Report:")
print(classification_report(true_labels, predictions, target_names=["negative", "positive"]))