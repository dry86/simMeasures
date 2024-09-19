import torch

def cal_fm_disagreement(tensor1, tensor2):
    # 获取两个tensor的长度，并取最小值，以较短的长度为准
    min_length = min(tensor1.size(1), tensor2.size(1))
    
    # 截取较短长度的tensor进行比较
    tensor1_trimmed = tensor1[0, :min_length]
    tensor2_trimmed = tensor2[0, :min_length]
    
    # 比较两个tensor对应位置的元素，统计不一致的个数
    disagreement_count = torch.sum(tensor1_trimmed != tensor2_trimmed).item()
    disagreement = disagreement_count / min_length
    
    return disagreement

# 示例数据
tensor1 = torch.tensor([[1, 515, 19229, 1053, 2391, 13, 13, 13, 1753, 756, 29918, 5358, 29918, 17664, 29898, 20326, 29901, 2391, 29961, 7411, 1053, 437, 312, 342, 13, 13, 1678, 437, 312, 342, 29889, 1688, 1545, 580, 13, 2]], device='cuda:1')
tensor2 = torch.tensor([[1, 515, 19229, 1053, 2391, 13, 13, 14, 1753, 756, 29918, 5358, 29918, 17664, 29898, 20326, 29901, 2391, 29961, 7411, 1053, 437, 312, 342, 13, 13, 1678, 437, 312, 342, 29889, 1688, 1545, 580, 13]], device='cuda:1')

# 计算不一致性
disagreement = cal_fm_disagreement(tensor1, tensor2)
print(f"Disagreement: {disagreement}")