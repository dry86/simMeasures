import numpy as np


def hsic(P, Q):
    PPt = P @ np.transpose(P)   # 计算线性核矩阵：K(X) = X * X.T
    QQt = Q @ np.transpose(Q)
    n = P.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n # H = I - (1/n) * 11.T
    
    hsic = np.trace(PPt @ H @ QQt @ H) / (n - 1) ** 2
    
    return hsic 

def rtd_cka(P, Q):
    """
    使用 Centered Kernel Alignment (CKA) 计算两个表示矩阵的相似性。
    先计算它们的线性核矩阵，然后进行中心化，并计算归一化的 HSIC
    """
    pp = np.sqrt(hsic(P, P) + 1e-10)
    qq = np.sqrt(hsic(Q, Q) + 1e-10)
    return hsic(P, Q) / pp / qq         # 归一化 HSIC 以获得 CKA 值

# 示例使用
if __name__ == "__main__":
    # 定义两个简单的表示
    R = np.array([[1, 0.5],
                  [0.5, 1],
                  [0.2, 0.8]])

    R_prime = np.array([[1, 0.4],
                        [0.4, 1],
                        [0.1, 0.9]])

    # 计算 CKA 相似度


    cka_score = rtd_cka(R, R_prime)
    print(f"CKA Similarity: {cka_score}")


