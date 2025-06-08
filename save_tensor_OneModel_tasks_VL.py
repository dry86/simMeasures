import os
import json5
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import jsonlines

def load_model_and_processor(model_path, device):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto"
    )
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def read_vl_prompts(data_path):
    prompts = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            # 每行为{"image_path":..., "text":...}
            prompts.append(obj)
    return prompts

def process_and_save_vl_hidden_states(model, processor, prompts, save_dir, device, batch_size=1, save_every=1000):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    accumulated_hidden_states = []
    batch_counter = 0
    batch_idx = 1
    for i, prompt in enumerate(prompts):
        image_path = prompt["image_path"]
        text = prompt["text"]
        image = Image.open(image_path).convert('RGB')
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            text=[text_prompt], images=[image], return_tensors="pt"
        )
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        hidden_states = outputs.hidden_states
        last_non_padding_index = inputs['attention_mask'].sum(dim=1) - 1
        last_token_hidden_states = [
            layer_output[torch.arange(layer_output.size(0)), last_non_padding_index, :].cpu()
            for layer_output in hidden_states
        ]
        accumulated_hidden_states.append(last_token_hidden_states)
        batch_counter += batch_size
        if batch_counter >= save_every:
            concatenated_hidden_states = torch.stack([torch.stack(states) for states in accumulated_hidden_states])
            concatenated_hidden_states = concatenated_hidden_states.permute(1, 0, 2)  # (num_layers, batch, hidden_size)
            torch.save(concatenated_hidden_states, f"{save_dir}vl_batch_{batch_idx}.pt")
            print(f"batch_{batch_idx} saved! shape: {concatenated_hidden_states.shape}")
            batch_idx += 1
            accumulated_hidden_states.clear()
            batch_counter = 0
    # 剩余未保存的
    if accumulated_hidden_states:
        concatenated_hidden_states = torch.stack([torch.stack(states) for states in accumulated_hidden_states])
        concatenated_hidden_states = concatenated_hidden_states.permute(1, 0, 2)
        torch.save(concatenated_hidden_states, f"{save_dir}vl_batch_{batch_idx}.pt")
        print(f"batch_{batch_idx} saved! shape: {concatenated_hidden_states.shape}")

def main(config_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    configs = json5.load(open(config_path))
    for config in configs:
        model_path = config["model_path"]
        data_path = config["data_path"]
        save_dir = config["save_dir"]
        print(f"Processing: model={model_path}, data={data_path}, save_dir={save_dir}")
        model, processor = load_model_and_processor(model_path, device)
        prompts = read_vl_prompts(data_path)
        process_and_save_vl_hidden_states(model, processor, prompts, save_dir, device)
        del model, processor
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main("config/config-save-tasks-vl.json5")

# # 获取每个样本最后一个有效(非padding)token的位置
# last_non_padding_index = inputs['attention_mask'].sum(dim=1) - 1  # shape: (batch,)

# # 获取每一层最后一个有效token的hidden state
# last_token_hidden_states = [
#     layer_output[torch.arange(layer_output.size(0)), last_non_padding_index, :].cpu()
#     for layer_output in hidden_states
# ]

# print("last_token_hidden_states (每层最后一个有效token的hidden state):")
# for i, h in enumerate(last_token_hidden_states):
#     print(f"Layer {i}: shape {h.shape}")

# # 你可以根据input_ids的分布，取你想要的token位置的表征
# # 例如，取第一个文本token的hidden state
# cls_token_hidden = last_token_hidden_states[-1]  # (batch, hidden_size)

# print("last_token_hidden_states:", last_token_hidden_states)
# print("--------------------------------")
# print("cls_token_hidden shape:", cls_token_hidden.shape)