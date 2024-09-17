import numpy as np
from numpy.linalg import norm, inv, eigvals
from scipy.linalg import sqrtm
from scipy.stats import pearsonr

def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度。
    """
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def calculate_rsm(representations):
    """
    计算给定表示的表征相似矩阵（RSM）。
    """
    n = representations.shape[0]  # 表示的数量
    rsm = np.zeros((n, n))
    
    # 计算每对表示之间的余弦相似度
    for i in range(n):
        for j in range(n):
            rsm[i, j] = cosine_similarity(representations[i], representations[j])
    
    # print(rsm)
    return rsm

# RSM Norm Difference
def frobenius_norm_diff(matrix1, matrix2):
    """
    计算两个矩阵之间差的Frobenius范数
    """
    diff = matrix1 - matrix2
    # print(diff)
    return norm(diff, 'fro')

def cal_RSM_Norm_Difference(rep1, rep2):
    """
    计算两个神经网络表示的RSM_Norm_Difference
    """
    # 计算每个表示的RSM
    rsm1 = calculate_rsm(rep1)
    rsm2 = calculate_rsm(rep2)
    
    # 计算RSM之间的Frobenius范数差异
    similarity = frobenius_norm_diff(rsm1, rsm2)
    
    return similarity

# RSA
def rsa_similarity(rsm1, rsm2):
    """
    使用皮尔逊相关系数计算两个RSM的向量化结果之间的相似性。
    """
    vec1 = rsm1[np.tril_indices(rsm1.shape[0], k=-1)]   # 将表征相似矩阵（RSM）的下三角部分展平为向量（不包括对角线）
    vec2 = rsm2[np.tril_indices(rsm2.shape[0], k=-1)]
    
    # 计算皮尔逊相关系数
    r, _ = pearsonr(vec1, vec2)
    return r

def cal_RSA(rep1, rep2):
    """
    使用RSA方法比较两个神经网络表示的相似性。
    """
    # 计算每个表示的RSM
    rsm1 = calculate_rsm(rep1)
    rsm2 = calculate_rsm(rep2)
    
    # 使用皮尔逊相关系数计算向量化RSM之间的相似性
    similarity = rsa_similarity(rsm1, rsm2)
    
    return similarity

# CKA
def hsic(P, Q):
    PPt = P @ np.transpose(P)   # 计算线性核矩阵：K(X) = X * X.T
    QQt = Q @ np.transpose(Q)
    n = P.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n # H = I - (1/n) * 11.T
    
    hsic = np.trace(PPt @ H @ QQt @ H) / (n - 1) ** 2   # 中心化和hsic计算
    
    return hsic 

def cal_cka(P, Q):
    """
    使用 Centered Kernel Alignment (CKA) 计算两个表示矩阵的相似性。
    先计算它们的线性核矩阵，然后进行中心化，并计算归一化的 HSIC
    """
    pp = np.sqrt(hsic(P, P) + 1e-10)
    qq = np.sqrt(hsic(Q, Q) + 1e-10)
    return hsic(P, Q) / pp / qq         # 归一化 HSIC 以获得 CKA 值

# Distance Correlation
def euclidean_distance_matrix(X):
    """
    计算给定表示矩阵的欧氏距离矩阵
    """
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))
    
    # 计算欧氏距离矩阵
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])
    
    return dist_matrix

def center_distance_matrix(D):
    """
    对距离矩阵进行中心化操作
    A_ij = D_ij - (1/n) * sum_k D_ik - (1/n) * sum_k D_kj + (1/n^2) * sum_kl D_kl
    """
    n = D.shape[0]
    row_mean = np.mean(D, axis=1)
    col_mean = np.mean(D, axis=0)
    total_mean = np.mean(D)
    
    # 中心化公式
    centered_D = D - row_mean[:, np.newaxis] - col_mean[np.newaxis, :] + total_mean
    return centered_D

def distance_covariance(X, Y):
    """
    计算两个表示的距离协方差
    """
    # 计算距离矩阵
    dist_X = euclidean_distance_matrix(X)
    dist_Y = euclidean_distance_matrix(Y)
    
    # 对距离矩阵进行中心化
    centered_X = center_distance_matrix(dist_X)
    centered_Y = center_distance_matrix(dist_Y)
    
    # 计算距离协方差
    dCov = np.sum(centered_X * centered_Y) / (X.shape[0] ** 2)
    return dCov

def cal_distance_correlation(X, Y):
    """
    计算两个表示矩阵的距离相关性
    """
    # 计算距离协方差
    dCov_XY = distance_covariance(X, Y)
    dCov_XX = distance_covariance(X, X)
    dCov_YY = distance_covariance(Y, Y)
    
    # 计算距离相关性
    dCor = dCov_XY / np.sqrt(dCov_XX * dCov_YY)
    return dCor

# Normalized Bures Similarity
def cal_bures_similarity(R, R_prime, epsilon=1e-6):
    """
    计算两个表示的 Normalized Bures Similarity (NBS)
    """
    K = R @ np.transpose(R) # 计算线性核
    K_prime = R_prime @ np.transpose(R_prime)

    # 正则化矩阵以保证数值稳定性
    K += epsilon * np.eye(K.shape[0])
    K_prime += epsilon * np.eye(K_prime.shape[0])

    # 计算矩阵的开方
    K_sqrt = sqrtm(K)
    
    # 计算 K_sqrt * K_prime * K_sqrt
    K_sqrt_K_prime = np.dot(K_sqrt, np.dot(K_prime, K_sqrt))
    
    # 计算 trace(sqrt(K_sqrt * K_prime * K_sqrt))
    trace_term = np.trace(sqrtm(K_sqrt_K_prime))
    
    # 计算 NBS 相似度
    nbs_similarity = trace_term / np.sqrt(np.trace(K) * np.trace(K_prime))
    
    return nbs_similarity

# Eigenspace Overlap Score
def cal_eigenspace_overlap_score(X, Y, top_k=None):
    """
    计算两个表示矩阵的 Eigenspace Overlap Score
    基于线性核来计算，并进行归一化。
    
    参数:
    X, Y: 两个表示矩阵
    top_k: 选择前 k 个特征向量进行比较，如果为 None，则比较所有特征向量
    """
    # 计算线性核矩阵
    K_X = X @ np.transpose(X) 
    K_Y = Y @ np.transpose(Y) 
    
    # 对线性核矩阵进行特征值分解
    eigvals_X, eigvecs_X = np.linalg.eigh(K_X)
    eigvals_Y, eigvecs_Y = np.linalg.eigh(K_Y)
    
    # 选择前 k 个特征向量进行比较
    if top_k is not None:
        eigvecs_X = eigvecs_X[:, -top_k:]  # 取前 k 个最大特征值对应的特征向量
        eigvecs_Y = eigvecs_Y[:, -top_k:]   
    
    # 计算特征向量之间的内积矩阵 (夹角)
    overlap_matrix = np.dot(eigvecs_X.T, eigvecs_Y)
    
    # 计算重叠得分：Frobenius 范数的平方
    overlap_score = np.linalg.norm(overlap_matrix, ord='fro')**2
    
    # 获取最大维度 D 和 D'，用于归一化
    max_rank = max(X.shape[0], Y.shape[0], X.shape[1], Y.shape[1])

    # 归一化得分
    normalized_overlap_score = overlap_score / max_rank
    
    return normalized_overlap_score

# GLUP
def cal_gulp_measure(R, R_prime, lambda_val=1e-3):
    """
    计算 GULP 度量 m^λ_GULP(R, R')
    
    参数:
    R: 表示矩阵 R (n_samples, n_features)
    R_prime: 表示矩阵 R' (n_samples, n_features)
    lambda_val: 岭回归的正则化参数 λ
    
    返回:
    GULP 度量
    """
    # 样本数量 N
    N = R.shape[0]

    # 计算协方差矩阵 S 和 S'
    S = (1 / N) * np.dot(R.T, R)
    S_prime = (1 / N) * np.dot(R_prime.T, R_prime)

    # 正则化的协方差矩阵 S^(-λ) 和 S'^(-λ)
    I_D = np.eye(S.shape[0])  # 单位矩阵
    S_inv_lambda = inv(S + lambda_val * I_D)
    S_prime_inv_lambda = inv(S_prime + lambda_val * I_D)

    # 计算跨表示的协方差矩阵 S_{R, R'}
    S_R_R_prime = (1 / N) * np.dot(R.T, R_prime)

    # 计算 GULP 度量
    term1 = np.trace(S_inv_lambda @ S @ S_inv_lambda @ S)
    term2 = np.trace(S_prime_inv_lambda @ S_prime @ S_prime_inv_lambda @ S_prime)
    term3 = 2 * np.trace(S_inv_lambda @ S_R_R_prime @ S_prime_inv_lambda @ S_R_R_prime.T)

    gulp_score = np.sqrt(term1 + term2 - term3)
    
    return gulp_score

def is_symmetric(matrix, tol=1e-8):
    """
    检查矩阵是否对称
    """
    return np.allclose(matrix, matrix.T, atol=tol)

def is_positive_definite(matrix):
    """
    检查矩阵是否正定
    """
    eigenvalues = eigvals(matrix)
    return np.all(eigenvalues > 0)

# Riemannian Distance
def cal_riemannian_distance(R, R_prime):
    """
    计算表示矩阵 R 和 R' 的 Riemannian Distance

    参数:
    R: 表示矩阵 R (n_samples, n_features)
    R_prime: 表示矩阵 R' (n_samples, n_features)

    返回:
    Riemannian Distance 度量
    """
    # 获取维度 D
    D = R.shape[1]
    
    # 计算协方差矩阵 S 和 S'
    S = np.dot(R, R.T) / D
    S_prime = np.dot(R_prime, R_prime.T) / D

    # 检查 S 和 S_prime 是否对称
    if not is_symmetric(S):
        print("Warning: Matrix S is not symmetric!")
        return -1
    if not is_symmetric(S_prime):
        print("Warning: Matrix S' is not symmetric!")
        return -1

    # 检查 S 和 S_prime 是否正定
    if not is_positive_definite(S):
        print("Warning: Matrix S is not positive definite!")
        return -1
    if not is_positive_definite(S_prime):
        print("Warning: Matrix S' is not positive definite!")
        return -1

    # 计算 S^(-1) S'
    S_inv = inv(S)
    S_inv_S_prime = np.dot(S_inv, S_prime)

    # 计算 S^(-1) S' 的特征值
    eigenvalues = eigvals(S_inv_S_prime)

    # 计算 Riemannian Distance
    log_eigenvalues = np.log(eigenvalues)
    riemannian_dist = np.sqrt(np.sum(log_eigenvalues**2))
    
    return riemannian_dist

# 示例使用
if __name__ == "__main__":
    # 定义两个简单的表示
    R = np.array([[1, 0.5],
                  [0.5, 1],
                  [0.2, 0.8]])

    R_prime = np.array([[1, 0.4],
                        [0.4, 1],
                        [0.1, 0.9]])
    
    # R_prime = np.array([[1, 0.5],
    #               [0.5, 1],
    #               [0.2, 0.8]])

    # 计算rsm相似度
    similarity_rsm = cal_RSM_Norm_Difference(R, R_prime)
    print(f"Representational Similarity (Frobenius norm difference): {similarity_rsm}")

    # 计算rsa相似度
    similarity_rsa = cal_RSA(R, R_prime)
    print(f"Representational Similarity (Pearson correlation): {similarity_rsa}")

    # 计算cka相似度
    similarity_cka = cal_cka(R, R_prime)
    print(f"CKA Similarity: {similarity_cka}")

    # 计算dCor距离相关性
    dcor_score = cal_distance_correlation(R, R_prime)
    print(f"Distance Correlation: {dcor_score}")

    # 计算 Normalized Bures Similarity
    nbs_score = cal_bures_similarity(R, R_prime)
    print(f"Normalized Bures Similarity: {np.real(nbs_score)}")

    # 计算 Eigenspace Overlap Score
    eos_score = cal_eigenspace_overlap_score(R, R_prime)
    print(f"Eigenspace Overlap Score (Normalized): {eos_score}")

    # 计算 GULP 度量
    gulp_score = cal_gulp_measure(R, R_prime)
    print(f"GULP Measure: {gulp_score:.4f}")

    # 计算 Riemannian Distance
    riemann_dist = cal_riemannian_distance(R, R_prime)
    print(f"Riemannian Distance: {riemann_dist:.4f}")
    