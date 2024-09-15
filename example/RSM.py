import numpy as np
from numpy.linalg import norm
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