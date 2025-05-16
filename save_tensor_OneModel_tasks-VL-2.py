import os
import json5
from PIL import Image
import torch
import jsonlines
from utils import get_model_class


def read_jsonl_prompts(data_path, image_base):
    prompts = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            image_path = os.path.join(image_base, obj["img"])
            text = obj["text"]
            problem = f"As a harmful-content detection expert, you are presented with a sample containing an image and the overlaid text \"{text}\". Considering both the visual and textual information, what is your judgment (harmful / harmless)?"
            prompts.append({"image_path": image_path, "problem": problem})
    return prompts

def save_hidden_states(batch_hidden_states, save_path, batch_idx):
    tensor = torch.stack([torch.stack(layer) for layer in batch_hidden_states])
    tensor = tensor.permute(1, 0, 2)  # (num_layers, batch, hidden_size)
    torch.save(tensor, os.path.join(save_path, f"vl_batch_{batch_idx}.pt"))
    print(f"Saved batch {batch_idx} to {save_path} with shape {tensor.shape}")


def main(config_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    configs = json5.load(open(config_path))
    
    for cfg in configs:
        model_type = cfg.get("model_type", "qwen_vl")  # e.g., qwen_vl
        model_path = cfg["model_path"]
        data_path = cfg["data_path"]
        image_base = cfg.get("image_base", "/newdisk/public/wws/00-Dataset-AIGC/FHM_new")
        save_dir = os.path.join(model_path, "pt_file", cfg.get("task", "vl"), cfg.get("lang", "en"))

        os.makedirs(save_dir, exist_ok=True)
        ModelClass = get_model_class(model_type)
        model = ModelClass(model_path, device)
        model.load_model()

        prompts = read_jsonl_prompts(data_path, image_base)

        accumulated = []
        batch_idx = 1
        for i, prompt in enumerate(prompts):
            hidden_states = model.get_hidden_states(prompt["image_path"], prompt["problem"])
            accumulated.append(hidden_states)

            if len(accumulated) >= 1000:
                save_hidden_states(accumulated, save_dir, batch_idx)
                batch_idx += 1
                accumulated.clear()
        
        if accumulated:
            save_hidden_states(accumulated, save_dir, batch_idx)

if __name__ == "__main__":
    main("/newdisk/public/wws/simMeasures/config/config-save-tasks-VL.json5")