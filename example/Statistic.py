import numpy as np
from numpy.linalg import norm

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
    
    normalized_var_mag = var_mag / (max_mag - min_mag)
    return normalized_var_mag

# 示例数据
R = np.array([[1, 2], [2, 3], [3, 4]])

# 执行
mag = magnitude(R)
var_mag = magnitude_variance(R)

print(f"Magnitude: {mag}")
print(f"Magnitude Variance: {var_mag}")


