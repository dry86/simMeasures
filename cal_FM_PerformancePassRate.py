import torch
import jsonlines
from utils import load_model_and_tokenizer
from human_eval.evaluation import evaluate_functional_correctness
from pathlib import Path
from tqdm import tqdm

# 定义函数用于生成代码并保存到本地
def generate_and_save_samples(model, tokenizer, file_path, output_file, num_samples=10):
    """
    使用模型生成代码，并将生成结果保存为 JSONL 格式。
    """
    with jsonlines.open(file_path) as reader, jsonlines.open(output_file, mode='w') as writer:
        for obj in tqdm(reader, desc="Processing tasks"):
            task_id = obj['task_id']
            prompt = obj['prompt']

            generated_samples = []
            for _ in range(num_samples):
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_length=512,
                        num_return_sequences=1,
                        do_sample=True,
                        top_k=50,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                completion = tokenizer.decode(generated_ids[0], skip_special_tokens=True)[len(prompt):]
                generated_samples.append(completion)

            for completion in generated_samples:
                writer.write({"task_id": task_id, "completion": completion})

# 主函数
def main():
    # 加载模型和tokenizer
    device = torch.device("cuda:3")  # 使用第0块GPU
    model_path = "/newdisk/public/wws/model_dir/deepseek-coder/dsc-7b-base-v1.5"
    model, tokenizer = load_model_and_tokenizer(model_path, device)

    # 定义文件路径
    problem_file = "/newdisk/public/wws/simMeasures/human_eval/HumanEval.jsonl"
    sample_file = "/newdisk/public/wws/simMeasures/human_eval/generated_samples_dsc100.jsonl"

    # 生成代码并保存到文件
    print("Generating samples...")
    generate_and_save_samples(
        model=model,
        tokenizer=tokenizer,
        file_path=problem_file,
        output_file=sample_file,
        num_samples=100  # 每个task生成10个样本
    )

    # 调用evaluate_functional_correctness进行评估
    print("Evaluating functional correctness...")
    pass_at_k_results = evaluate_functional_correctness(
        sample_file=sample_file,
        k=[1, 10, 100], # k = 1, 5, 10
        n_workers=4,
        timeout=3.0,
        problem_file=problem_file
    )

    # 打印结果
    print("Pass@k results:")
    for k, score in pass_at_k_results.items():
        print(f"{k}: {score:.4f}")

if __name__ == "__main__":
    main()
