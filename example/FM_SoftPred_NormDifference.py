import numpy as np

def norm_of_soft_prediction_diff(O, O_prime):
    """
    计算两个输出O和O'之间的Norm of Soft Prediction Difference
    O和O'为两个模型的输出，形状为(N, C)，其中N是实例数，C是类数
    """
    N = O.shape[0]
    
    # 计算每个实例对应的欧几里得距离
    distances = np.linalg.norm(O - O_prime, axis=1)
    
    # 计算平均差异
    m_pred_norm_diff = np.sum(distances) / (2 * N)
    
    return m_pred_norm_diff

# 示例模型输出
O = np.array([[0.1, 0.9], [0.3, 0.7], [0.4, 0.6]])
O_prime = np.array([[0.2, 0.8], [0.25, 0.75], [0.5, 0.5]])

# 计算Norm of Soft Prediction Difference
result = norm_of_soft_prediction_diff(O, O_prime)
print(f"Norm of Soft Prediction Difference: {result}")