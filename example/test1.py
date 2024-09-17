import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 示例数据：假设有3个实例表示向量，每个向量有2个特征
R = np.array([[1, 2], [3, 4], [5, 6]])

# 计算表示的均值
mean_R = np.mean(R, axis=0)

# 计算每个实例与均值的余弦相似度
def concentricity(R):
    mean_R = np.mean(R, axis=0)
    cos_sims = []
    for r in R:
        cos_sim = cosine_similarity([r], [mean_R])[0][0]
        cos_sims.append(cos_sim)
    return np.array(cos_sims)

# 计算Concentricity均值
def mean_concentricity(R):
    cos_sims = concentricity(R)
    return np.mean(cos_sims)

# 计算Concentricity Variance
def concentricity_variance(R):
    cos_sims = concentricity(R)
    mean_cos_sim = np.mean(cos_sims)
    var_conc = np.sqrt(np.mean(cos_sims - mean_cos_sim))
    max_cos_sim = np.max(cos_sims)
    min_cos_sim = np.min(cos_sims)
    
    # 使用归一化的方差公式
    normalized_var_conc = var_conc / (max_cos_sim - min_cos_sim) if max_cos_sim != min_cos_sim else 0
    return normalized_var_conc

# 执行
mean_conc = mean_concentricity(R)
var_conc = concentricity_variance(R)

print(f"Mean Concentricity: {mean_conc}")
print(f"Concentricity Variance: {var_conc}")