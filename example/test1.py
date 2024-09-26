import numpy as np

# 假设一组矩阵 R_list
R_list = [
    np.array([[1, 2], [3, 4]]),
    np.array([[2, 0], [1, 1]]),
    np.array([[0, 1], [4, 5]])
]

# Frobenius 范数
def frobenius_norm(matrix):
    return np.linalg.norm(matrix, 'fro')


# 平均幅度
def mean_magnitude(R_list):
    magnitudes = [frobenius_norm(R) for R in R_list]  # 计算每个矩阵的 Frobenius 范数
    return np.mean(magnitudes)


mean_mag = mean_magnitude(R_list)

print(f"Mean Magnitude: {mean_mag}")