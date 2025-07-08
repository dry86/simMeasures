import os
import io
import json5
from PIL import Image
import torch
import jsonlines
from utils import get_model_class
from datasets import load_dataset
from tqdm import tqdm

def process_FHM_prompts():
    data_path = "/newdisk/public/wws/00-Dataset-AIGC/FHM_new/test.jsonl"
    image_base = "/newdisk/public/wws/00-Dataset-AIGC/FHM_new"
    prompts = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            image_path = os.path.join(image_base, obj["img"])
            image = Image.open(image_path).convert("RGB")
            text = obj["text"]
            text = f"As a hateful-content detection expert, you are presented with a sample containing an image and the overlaid text \"{text}\". Considering both the visual and textual information, what is your judgment (hateful / non-hateful)?"
            prompts.append({"image": image, "text": text})
    return prompts

def process_VQAv2_prompts():
    data_path = "/newdisk/public/wws/simMeasures/VL_Task/VQAv2_sample_test/data/test-00000-of-00001.parquet"
    prompts = []
    dataset = load_dataset("parquet", data_files=data_path, split="train")
    for obj in tqdm(dataset):
        image_path = obj["image"]["bytes"]
        image = Image.open(io.BytesIO(image_path)).convert("RGB")
        text = obj["question"]
        prompts.append({"image": image, "text": text})
    return prompts

def process_flickr30k_prompts():
    data_path = "/newdisk/public/wws/simMeasures/VL_Task/flickr30k_test/data/test-00000-of-00001.parquet"
    prompts = []
    dataset = load_dataset("parquet", data_files=data_path, split="train")
    for obj in tqdm(dataset):
        image = obj["image"]
        # image = Image.open(io.BytesIO(image_path)).convert("RGB")
        text = "Describe this image in one sentence."
        prompts.append({"image": image, "text": text})
    return prompts

def save_hidden_states(batch_hidden_states, save_path, batch_idx):
    tensor = torch.stack([torch.stack(layer) for layer in batch_hidden_states])
    tensor = tensor.permute(1, 0, 2)  # (num_layers, batch, hidden_size)
    torch.save(tensor, os.path.join(save_path, f"vl_batch_{batch_idx}.pt"))
    print(f"Saved batch {batch_idx} to {save_path} with shape {tensor.shape}")


def main(config_path):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    configs = json5.load(open(config_path))
    
    for cfg in configs:
        model_type = cfg.get("model_type", "none")  # e.g., qwen_vl
        model_path = cfg["model_path"]
        tasks = cfg.get("task")

        print(model_type, model_path, tasks)
        
    print("-"*50)

    for cfg in configs:
        model_type = cfg.get("model_type", "none")  # e.g., qwen_vl
        model_path = cfg["model_path"]
        model_name = os.path.basename(model_path)
        tasks = cfg.get("task")
        
        # 确保tasks是列表
        if isinstance(tasks, str):
            tasks = [tasks]
            
        ModelClass = get_model_class(model_type)
        model = ModelClass(model_path, device)
        model.load_model()
        
        for task in tasks:
            save_dir = os.path.join("/newdisk/public/wws/simMeasures/VL_Model", model_name, "pt_file", task)
            os.makedirs(save_dir, exist_ok=True)
            
            print(f"Processing task: {task} for model: {model_name}")
            
            if task == "FHM":
                prompts = process_FHM_prompts()
            elif task == "VQAv2":
                prompts = process_VQAv2_prompts()
            elif task == "flickr30k":
                prompts = process_flickr30k_prompts()
            else:
                print(f"Unsupported task: {task}, skipping...")
                continue

            accumulated = []
            batch_idx = 1
            for i, prompt in tqdm(enumerate(prompts)):
                hidden_states = model.get_hidden_states(prompt["image"], prompt["text"])
                accumulated.append(hidden_states)

                if len(accumulated) >= 1000:
                    save_hidden_states(accumulated, save_dir, batch_idx)
                    batch_idx += 1
                    accumulated.clear()
            
            if accumulated:
                save_hidden_states(accumulated, save_dir, batch_idx)
            
            print(f"Completed task: {task} for model: {model_name}")

if __name__ == "__main__":
    main("/newdisk/public/wws/simMeasures/config/config-save-tasks-VL.json5")