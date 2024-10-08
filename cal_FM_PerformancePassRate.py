import torch
import jsonlines
from utils import load_model_and_tokenizer
from evaluate import load
from human_eval import evaluate_functional_correctness

# Load pass@k metric
pass_at_k_metric = load("pass_at_k")

def generate_code(prompt, model, tokenizer, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_return_sequences=1,  # Adjust for different number of sequences if necessary
            do_sample=True,          # Enable sampling for diverse outputs
            top_k=50,                # Control diversity with top-k sampling
            temperature=0.7          # Adjust temperature for creativity
        )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

# Example of calculating pass@k
def calculate_pass_at_k(model, tokenizer, prompt, canonical_solution, test, k=1, num_samples=100):
    generated_codes = [generate_code(prompt, model, tokenizer) for _ in range(num_samples)]

    # Evaluate generated codes against test case (you'll need to run the generated code and see if it passes)
    results = []
    for code in generated_codes:
        try:
            exec_globals = {}
            exec(code, exec_globals)  # Execute generated code
            exec(test, exec_globals)  # Execute test case
            result = exec_globals.get('test_result', False)
            results.append(result)
        except Exception as e:
            results.append(False)

    # Calculate pass@k score using `evaluate`
    return pass_at_k_metric.compute(references=[canonical_solution], predictions=generated_codes, num_samples=num_samples, k=k)

def main(model_1, model_2, file_path, device1, device2):

    model1, tokenizer1 = load_model_and_tokenizer(model_1, device1)
    # model2, tokenizer2 = load_model_and_tokenizer(model_2, device2)


    with jsonlines.open(file_path) as reader:
        for obj in reader:
            task_id = obj.get('task_id')
            prompt = obj.get('prompt')
            canonical_solution = obj.get('canonical_solution')
            test = obj.get('test')
            
            pass_k_score = calculate_pass_at_k(model1, tokenizer1, prompt, canonical_solution, test, k=1, num_samples=1)
            
            print(f"Task: {task_id} - pass@1 score: {pass_k_score}")

if __name__ == "__main__":


    # 指定GPU设备：
    device_model1 = torch.device("cuda:0")  # 第x块GPU
    device_model2 = torch.device("cuda:1")  # 第y块GPU

    device_model1 = 'cpu'
    device_model2 = 'cpu'

    # 设置模型和输入
    model_1 = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b"
    model_2 = "/newdisk/public/wws/model_dir/codellama/CodeLlama-7b-Instruct-hf" # "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b-Python"

    # 打开jsonl文件并遍历
    file_path = "/newdisk/public/wws/simMeasures/human-eval-master/data/HumanEval.jsonl"  # Dataset

    main(model_1, model_2, file_path, device_model1, device_model2)