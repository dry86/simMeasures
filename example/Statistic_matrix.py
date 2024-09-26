import numpy as np
from numpy.linalg import norm


# Frobenius 范数
def frobenius_norm(matrix):
    return np.linalg.norm(matrix)

# 平均幅度
def mean_magnitude(R_list):
    magnitudes = [frobenius_norm(R) for R in R_list]  # 计算每个矩阵的 Frobenius 范数
    return np.mean(magnitudes)

# 幅度方差
def magnitude_variance(R_list):
    magnitudes = np.array([frobenius_norm(R) for R in R_list])  # 每个矩阵的 Frobenius 范数
    max_mag = np.max(magnitudes)
    min_mag = np.min(magnitudes)
    mean_mag = np.mean(magnitudes)
    
    variance = np.sqrt(np.mean((magnitudes - mean_mag) ** 2))
    normalized_variance =  variance  /  (max_mag - min_mag)
    
    return normalized_variance

# Concentricity
# 余弦相似度函数
def cosine_similarity(A, B):
    return np.dot(A.flatten(), B.flatten()) / (norm(A) * norm(B))

# 计算同心度
def concentricity(R_list):
    R_mean = np.mean(R_list, axis=0)  # 计算平均矩阵
    alpha_list = [cosine_similarity(R, R_mean) for R in R_list]  # 每个矩阵和平均矩阵之间的余弦相似度
    return alpha_list

# 平均同心度
def mean_concentricity(R_list):
    alpha_list = concentricity(R_list)
    return np.mean(alpha_list)

# 同心度方差
def concentricity_variance(R_list):
    alpha_list = concentricity(R_list)
    max_alpha = np.max(alpha_list)
    min_alpha = np.min(alpha_list)
    mean_alpha = np.mean(alpha_list)
    
    variance = np.sqrt(np.mean((alpha_list - mean_alpha) ** 2))
    normalized_variance = variance / (max_alpha - min_alpha) 
    
    return normalized_variance


# 计算矩阵之间的欧氏距离的平方
def squared_euclidean_distance(A, B):
    dist = np.sum((A - B) ** 2)
    return dist

# 通过欧几里德距离计算均匀性
def uniformity_euclidean(R_list):
    N = len(R_list)
    total_sum = 0
    t = 1 / R_list[0].shape[1]    # t = 1 / 特征数
    # 计算每对矩阵之间的距离并使用高斯核
    for i in range(N):
        for j in range(N):
            distance_squared = squared_euclidean_distance(R_list[i], R_list[j])
            total_sum += np.exp(-t * distance_squared)
    
    # 计算均匀性度量
    uniformity_value = np.log(total_sum / (N ** 2))
    
    return uniformity_value

# 计算矩阵的逐元素内积（Frobenius 内积）
def frobenius_inner_product(A, B):
    return np.sum(A * B)

# 通过矩阵内积计算均匀性
def uniformity_inner(R_list):
    N = len(R_list)
    total_sum = 0
    t = 1 / R_list[0].shape[1]  # t = 1 / 特征数
    # 计算每对矩阵之间的距离并使用高斯核
    for i in range(N):
        for j in range(N):
            inner_product = frobenius_inner_product(R_list[i], R_list[j])
            total_sum += np.exp(2 * t * inner_product - 2 * t)
    
    # 计算均匀性度量
    uniformity_value = np.log(total_sum / (N ** 2))
    
    return uniformity_value


# 示例使用
if __name__ == "__main__":

    # 假设一组矩阵 R_list
    R_list = [
        np.array([[1, 2], [3, 4]]),
        np.array([[2, 0], [1, 1]]),
        np.array([[0, 1], [4, 5]])
    ]

    # 执行
    mean_mag = mean_magnitude(R_list)
    var_mag = magnitude_variance(R_list)

    print(f"Mean Magnitude: {mean_mag}")
    print(f"Magnitude Variance: {var_mag}")


    mean_conc = mean_concentricity(R_list)
    var_conc = concentricity_variance(R_list)

    print(f"Mean Concentricity: {mean_conc}")
    print(f"Concentricity Variance: {var_conc}")


    # 调整 t 值并查看 uniformity_value 的变化
    # for t in [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]:
    uniformity_value = uniformity_euclidean(R_list)
    print(f"Uniformity_euclidean: {uniformity_value}")


    uniformity_value = uniformity_inner(R_list)
    print(f"Uniformity_inner: {uniformity_value}")