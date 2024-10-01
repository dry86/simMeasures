import torch
from getHiddenStates import load_hidden_states, concatenate_hidden_states
import numpy as np
from example import cca_core
from tqdm import tqdm
from repsim.measures import *
from repsim.measures.cca import get_cca_similarity
from utils import print_and_save

def cca_decomp(A, B, device):
    """Computes CCA vectors, correlations, and transformed matrices
    requires a < n and b < n
    Args:
        A: np.array of size a x n where a is the number of neurons and n is the dataset size
        B: np.array of size b x n where b is the number of neurons and n is the dataset size
    Returns:
        u: left singular vectors for the inner SVD problem
        s: canonical correlation coefficients
        vh: right singular vectors for the inner SVD problem
        transformed_a: canonical vectors for matrix A, a x n array
        transformed_b: canonical vectors for matrix B, b x n array
    """
    # Move A and B to the specified device
    A = A.to(device)
    B = B.to(device)
    assert A.shape[0] < A.shape[1]
    assert B.shape[0] < B.shape[1]

    evals_a, evecs_a = torch.linalg.eigh(A @ A.T)
    evals_a = (evals_a + torch.abs(evals_a)) / 2
    inv_a = torch.tensor([1 / torch.sqrt(x) if x > 0 else 0 for x in evals_a], device=device)

    evals_b, evecs_b = torch.linalg.eigh(B @ B.T)
    evals_b = (evals_b + torch.abs(evals_b)) / 2
    inv_b = torch.tensor([1 / torch.sqrt(x) if x > 0 else 0 for x in evals_b], device=device)

    cov_ab = A @ B.T

    temp = (
        (evecs_a @ torch.diag(inv_a) @ evecs_a.T)
        @ cov_ab
        @ (evecs_b @ torch.diag(inv_b) @ evecs_b.T)
    )

    try:
        u, s, vh = torch.linalg.svd(temp)
    except:
        u, s, vh = torch.linalg.svd(temp * 100)
        s = s / 100

    transformed_a = (u.T @ (evecs_a @ torch.diag(inv_a) @ evecs_a.T) @ A).T
    transformed_b = (vh @ (evecs_b @ torch.diag(inv_b) @ evecs_b.T) @ B).T
    return u, s, vh, transformed_a, transformed_b

def mean_sq_cca_corr(rho):
    """Compute mean squared CCA correlation
    :param rho: canonical correlation coefficients returned by cca_decomp(A,B)
    """
    # len(rho) is min(A.shape[0], B.shape[0])
    return torch.sum(rho * rho) / len(rho)

def mean_cca_corr(rho):
    """Compute mean CCA correlation
    :param rho: canonical correlation coefficients returned by cca_decomp(A,B)
    """
    # len(rho) is min(A.shape[0], B.shape[0])
    return torch.sum(rho) / len(rho)


def cal_resi_cca(acts1, acts2, row, sheet):

    shape = "nd"

    score = get_cca_similarity(
        acts1.T,
        acts2.T,
        epsilon=1e-8,
        compute_dirns=False,
        compute_coefs=True,
        verbose=False,
    )
    # print("\t CCA epsilon=1e-8: ", np.mean(score["cca_coef1"]))
    print_and_save("resiCCA", np.mean(score["cca_coef1"]), row=row, sheet=sheet)

    svcca = SVCCA()
    score = svcca(acts1, acts2, shape)
    # print("\t resi SVCCA: ", score)
    print_and_save("resiSVCCA", score, row=row, sheet=sheet)

    pwcca = PWCCA()
    score = pwcca(acts1, acts2, shape)
    # print("\t resi PWCCA: ", score)
    print_and_save("resiPWCCA", score, row=row, sheet=sheet)


def calculate_cca(acts1, acts2, idx, sheet):
    # acts1 = acts1.T # convert to neurons by datapoints
    # acts2 = acts2.T
    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")
    results = cca_core.get_cca_similarity(acts1, acts2, epsilon=1e-8, verbose=False)

    # print(f"\t Mean CCA similarity: {np.mean(results["cca_coef1"])}")
    print_and_save("MeanCCA", np.mean(results["cca_coef1"]), row=idx, sheet=sheet)

    svcca_res = cca_core.compute_svcca(acts1, acts2)
    # print("\t SVCCA similarity: ", svcca_res)
    print_and_save("SVCCA", svcca_res, row=idx, sheet=sheet)

    pwcca_mean, w, _ = cca_core.compute_pwcca(results, acts1, acts2)
    # print("\t PWCCA similarity: ", pwcca_mean)
    print_and_save("PWCCA", pwcca_mean, row=idx, sheet=sheet)


def main(model1_path, model2_path, device1, device2):
    """主函数：加载模型、读取数据、计算CCA相似性"""
    lang_sheet = model1_path.split('/')[-1] # 拿到模型对比的数据集的语言, 在写入时作为sheet名称

    # 获取隐藏层输出
    hidden_states_model1 = concatenate_hidden_states(model1_path, "hsm1", device1)
    hidden_states_model2 = concatenate_hidden_states(model2_path, "hsm2", device2)

    # 获取模型的总层数并计算每一层的CCA相关性得分
    num_layers = len(hidden_states_model1)
    for i in tqdm(range(num_layers)):

        # 先将每层所有数据的隐藏层激活拼接成三维矩阵 (batch_size, max_length, hidden_size)
        layer_activations_model1 = hidden_states_model1[i]  # 形状 (batch_size, max_length, hidden_size)
        layer_activations_model2 = hidden_states_model2[i]  # 形状 (batch_size, max_length, hidden_size)

        # 通过 view() 函数将其变成二维矩阵 (batch_size * max_length, hidden_size)
        acts1 = layer_activations_model1.view(-1, layer_activations_model1.shape[-1])
        acts2 = layer_activations_model2.view(-1, layer_activations_model2.shape[-1])

        device = acts1.device  # 获取 acts1 所在设备
        acts2_device = acts2.to(device)  # 将 acts2 移动到 acts1 所在的设备
        print(f"Layer {i}, acts1 shape: {acts1.shape}:")
        cka = CKA()
        score = cka(acts1, acts2_device, "nd")
        # print("\t CKA: ", score)
        print_and_save("CKA", score, row=i, sheet=lang_sheet)

        cal_resi_cca(acts1.cpu().numpy(), acts2.cpu().numpy(), i, lang_sheet)

        acts1 = acts1.T # convert to neurons by datapoints
        acts2 = acts2.T

        acts1_numpy = acts1.cpu().numpy()
        acts2_numpy = acts2.cpu().numpy()

        # 计算该层的CCA
        calculate_cca(acts1_numpy, acts2_numpy, i, lang_sheet)

if __name__ == "__main__":

    device_model1 = torch.device("cuda:2")  # 第x块GPU
    device_model2 = torch.device("cuda:3")  # 第y块GPU

    # 模型和数据路径
    pt_model_7b = "/newdisk/public/wws/simMeasures/pt_file/Python"
    pt_model_7b_Python = "/newdisk/public/wws/simMeasures/pt_file/Python"
    
    # 调用主函数
    main(pt_model_7b, pt_model_7b_Python, device_model1, device_model2)
            


