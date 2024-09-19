import numpy as np
from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度。
    """
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Magnitude 计算
def magnitude(R):
    mean_R = np.mean(R, axis=0)
    mag = norm(mean_R)
    return mag

# Magnitude Variance 计算
def magnitude_variance(R):
    magnitudes = norm(R, axis=1)
    mean_mag = np.mean(magnitudes)
    var_mag = np.sqrt(np.mean((magnitudes - mean_mag) ** 2))
    max_mag = np.max(magnitudes)
    min_mag = np.min(magnitudes)
    
    normalized_var_mag = var_mag / (max_mag - min_mag)  if max_mag != min_mag else 0
    return normalized_var_mag

# Concentricity
# 计算每个实例与均值的余弦相似度
def concentricity(R):
    mean_R = np.mean(R, axis=0)
    cos_sims = []
    for r in R:
        cos_sim = cosine_similarity(r, mean_R)
        cos_sims.append(cos_sim)
    return np.array(cos_sims)

# 计算Concentricity均值
def mean_concentricity(R):
    cos_sims = concentricity(R)
    return np.mean(cos_sims)

# 计算Concentricity方差
def concentricity_variance(R):
    cos_sims = concentricity(R)
    mean_cos_sim = np.mean(cos_sims)    # Todo: cos_sims 是否要加 2范数 ? Magnitude 同
    var_conc = np.sqrt(np.mean((cos_sims - mean_cos_sim) ** 2))
    max_cos_sim = np.max(cos_sims)
    min_cos_sim = np.min(cos_sims)
    
    # 使用归一化的方差公式
    normalized_var_conc = var_conc / (max_cos_sim - min_cos_sim) if max_cos_sim != min_cos_sim else 0
    return normalized_var_conc

# 计算Uniformity
def uniformity(R, t=2):
    N = R.shape[0]
    # 计算每对实例的欧氏距离
    dist_matrix = np.linalg.norm(R[:, np.newaxis] - R[np.newaxis, :], axis=-1) ** 2
    # 计算公式中的指数项
    exp_term = np.exp(-t * dist_matrix)
    # 计算uniformity值
    uniformity_value = np.log(np.sum(exp_term) / (N ** 2))
    return uniformity_value

# 计算Uniformity
def uniformity2(R, t=2):
    N = R.shape[0]
    # 计算每对实例的欧氏距离
    dist_matrix = np.linalg.norm(R[:, np.newaxis] - R[np.newaxis, :], axis=-1) ** 2
    # 计算公式中的指数项
    exp_term = np.exp(-t * dist_matrix)
    # 计算uniformity值
    uniformity_value = np.log(np.sum(exp_term) / (N ** 2))
    return uniformity_value


# 示例使用
if __name__ == "__main__":

    # 示例数据
    R = np.array([[1, 2], [2, 3], [3, 4]])

    # 执行
    mag = magnitude(R)
    var_mag = magnitude_variance(R)

    print(f"Magnitude: {mag}")
    print(f"Magnitude Variance: {var_mag}")

    # 示例数据：假设有3个实例表示向量，每个向量有2个特征
    R = np.array([[1, 2], [3, 4], [5, 6]])
    # 执行
    mean_conc = mean_concentricity(R)
    var_conc = concentricity_variance(R)

    print(f"Mean Concentricity: {mean_conc}")
    print(f"Concentricity Variance: {var_conc}")

    uniformity_value = uniformity(R, t=-0.01)
    print(f"Uniformity: {uniformity_value}")