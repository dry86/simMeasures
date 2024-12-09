from transformers import AutoModelForCausalLM
import torch
# from torchinfo import summary  # Optional: Ensure torchinfo is installed


# 1. 加载模型
def load_model(model_name_or_path, device):
    """
    Load a Hugging Face model using AutoModelForCausalLM.
    Args:
        model_name_or_path (str): Path to the local model or Hugging Face model name.
    Returns:
        model: Loaded model.
    """
    return AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)


# 2. 查看模型结构
def print_model_structure(model):
    """
    Print the architecture of the model.
    Args:
        model: Loaded Hugging Face model.
    """
    print(model)


# 3. 列出模型参数及其形状
def list_model_parameters(model):
    """
    List all parameter names and their shapes in the model.
    Args:
        model: Loaded Hugging Face model.
    Returns:
        param_info (list): List of tuples (parameter_name, parameter_shape).
    """
    param_info = [(name, param.shape) for name, param in model.named_parameters()]
    for name, shape in param_info:
        print(f"Parameter Name: {name}, Shape: {shape}")
    return param_info


# 4. 统计模型总参数量
def calculate_total_parameters(model):
    """
    Calculate the total number of parameters in the model.
    Args:
        model: Loaded Hugging Face model.
    Returns:
        total_params (int): Total number of parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    return total_params


# 5. 提取并打印特定层的参数
def extract_specific_layer_parameters(model, layer_keyword):
    """
    Extract parameters for a specific layer based on a keyword in the layer name.
    Args:
        model: Loaded Hugging Face model.
        layer_keyword (str): Keyword to filter layers (e.g., "layer.0").
    Returns:
        specific_params (list): List of tuples (parameter_name, parameter_shape).
    """
    specific_params = [(name, param.shape) for name, param in model.named_parameters() if layer_keyword in name]
    for name, shape in specific_params:
        print(f"Parameter Name: {name}, Shape: {shape}")
    return specific_params


# 6. 打印模型详细信息
def summarize_model(model, input_size):
    """
    Summarize the model using torchinfo.
    Args:
        model: Loaded Hugging Face model.
        input_size (tuple): Input size (batch_size, seq_length).
    """
    print(summary(model, input_size=input_size))


# 7. 对比两个模型的参数差异
def compare_model_parameters(model1, model2):
    """
    Compare the parameters of two models and print their differences.
    Args:
        model1: First Hugging Face model.
        model2: Second Hugging Face model.
    """
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 == name2:
            difference = torch.sum(param1 - param2).item()
            print(f"Layer {name1}: Difference {difference}")


# 8. 保存模型参数到文件
def save_model_parameters_to_file(model, file_path):
    """
    Save the model parameters to a file (e.g., for further analysis).
    Args:
        model: Loaded Hugging Face model.
        file_path (str): Path to save the model parameters.
    """
    state_dict = model.state_dict()
    torch.save(state_dict, file_path)
    print(f"Model parameters saved to {file_path}")

if __name__ == "__main__":

    device_model = torch.device("cuda:1")

    model_list = ["/newdisk/public/wws/model_dir/codellama/codeLlama-7b",
                  "/newdisk/public/wws/model_dir/codellama/codeLlama-7b-Instruct",
                  "/newdisk/public/wws/model_dir/deepseek-coder/dsc-6.7b-base",
                "/newdisk/public/wws/model_dir/deepseek-coder/dsc-7b-base-v1.5",
                  "/newdisk/public/wws/model_dir/StarCoder2/starcoder2-7b",
                  "/newdisk/public/wws/model_dir/Qwen2.5-Coder/Qwen2.5-Coder-7B"]
    
    for model_path in model_list:
        model_path = "/newdisk/public/wws/model_dir/MagiCoder/magicoder-CL-7b"
        print(f"model_path: {model_path}")
        model = load_model(model_path, device_model)

        print_model_structure(model)    # 总体的模型结构

        # params = list_model_parameters(model)   # 每层具体的参数

        total_params = calculate_total_parameters(model)

        # layer_params = extract_specific_layer_parameters(model, "layer.0")
        break
