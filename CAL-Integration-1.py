import torch
from getHiddenStates import concatenate_hidden_states
import numpy as np
from example import cca_core
from tqdm import tqdm
from utils import print_and_save

from repsim.measures.procrustes import orthogonal_procrustes
from repsim.measures.nearest_neighbor import joint_rank_jaccard_similarity
from repsim.measures import *

def cal_Statistic(acts1, acts2, shape, idx, sheet):

    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")

    difference = MagnitudeDifference()
    score = difference(acts1, acts2, shape)
    print_and_save("MagDiff", score, idx, sheet)

    difference = ConcentricityDifference()
    score = difference(acts1, acts2, shape)
    print_and_save("ConDiff", score, idx, sheet)

    # difference = UniformityDifference()
    # score = difference(acts1, acts2, shape)
    # print_and_save("UniDiff", score, idx, sheet)

def cal_Neighbors(acts1, acts2, shape, idx, sheet):

    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")

    jaccard = JaccardSimilarity()
    score = jaccard(acts1, acts2, shape)
    print_and_save("JacSim", score, idx, sheet)

    secondOrder_cosine = SecondOrderCosineSimilarity()
    score = secondOrder_cosine(acts1, acts2, shape)
    print_and_save("SecOrdCosSim", score, idx, sheet)

    rankSim = RankSimilarity()
    score = rankSim(acts1, acts2, shape)
    print_and_save("RankSim", score, idx, sheet)

    score = joint_rank_jaccard_similarity(acts1, acts2, shape)
    print_and_save("RankJacSim", score, idx, sheet)

def cal_RSM(acts1, acts2, shape, idx, sheet):

    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")

    # 计算耗时太长 !
    # rsm_norm_diff = RSMNormDifference()
    # score = rsm_norm_diff(acts1, acts2, shape)
    # print_and_save("RSMNormDiff", score, idx, sheet)

    rsa = RSA()
    score = rsa(acts1, acts2, shape)
    print_and_save("RSA", score, idx, sheet)

    cka = CKA()
    score = cka(acts1, acts2, shape)
    print_and_save("CKA", score, idx, sheet)

    dCor = DistanceCorrelation()
    score = dCor(acts1, acts2, shape)
    print_and_save("DisCor", score, idx, sheet)

    eigenspace_overlap = EigenspaceOverlapScore()
    score = eigenspace_overlap(acts1, acts2, shape)
    print_and_save("EOlapScore", score, idx, sheet)

    gulp = Gulp()
    score = gulp(acts1, acts2, shape)
    print_and_save("Gulp", score, idx, sheet)

def cal_Alignment(acts1, acts2, shape, idx, sheet):

    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")

    score = orthogonal_procrustes(acts1, acts2, shape)
    print_and_save("OrthPro", score, idx, sheet)

    opcan = OrthogonalProcrustesCenteredAndNormalized()
    score = opcan(acts1, acts2, shape)
    print_and_save("OrthProCAN", score, idx, sheet)

    linear_regression = LinearRegression()
    score = linear_regression(acts1, acts2, shape)
    print_and_save("LinRegre", score, idx, sheet)

    aligned_cosine_sim = AlignedCosineSimilarity()
    score = aligned_cosine_sim(acts1, acts2, shape)
    print_and_save("AliCosSim", score, idx, sheet)

    soft_correlation_match = SoftCorrelationMatch()
    score = soft_correlation_match(acts1, acts2, shape)
    print_and_save("SoftCorMatch", score, idx, sheet)

    hard_correlation_match = HardCorrelationMatch()
    score = hard_correlation_match(acts1, acts2, shape)
    print_and_save("HardCorMatch", score, idx, sheet)

def calculate_cca(acts1, acts2, idx, sheet):

    acts1 = acts1.T # convert to neurons by datapoints
    acts2 = acts2.T
    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")
    results = cca_core.get_cca_similarity(acts1, acts2, epsilon=1e-8, verbose=False)
    print_and_save("MeanCCA", np.mean(results["cca_coef1"]), row=idx, sheet=sheet)

    svcca_res = cca_core.compute_svcca(acts1, acts2)
    print_and_save("SVCCA", svcca_res, row=idx, sheet=sheet)

    pwcca_mean, w, _ = cca_core.compute_pwcca(results, acts1, acts2)
    print_and_save("PWCCA", pwcca_mean, row=idx, sheet=sheet)


def main(model1_path, model2_path, device1, device2):
    """主函数：加载模型、读取数据、计算CCA相似性"""
    lang_sheet = model1_path.split('/')[-1] # 拿到模型对比的数据集的语言, 在写入时作为sheet名称

    # 获取隐藏层输出
    hidden_states_model1 = concatenate_hidden_states(model1_path, "7b", device1)
    hidden_states_model2 = concatenate_hidden_states(model2_path, "7bInstruct", device2)

    # 获取模型的总层数并计算每一层的CCA相关性得分
    num_layers = len(hidden_states_model1)
    for i in tqdm(range(num_layers)):

        # if i < 16:
        #     continue

        # 先将每层所有数据的隐藏层激活拼接成三维矩阵 (batch_size, max_length, hidden_size)
        layer_activations_model1 = hidden_states_model1[i]  # 形状 (batch_size, max_length, hidden_size)
        layer_activations_model2 = hidden_states_model2[i]  # 形状 (batch_size, max_length, hidden_size)

        # 通过 view() 函数将其变成二维矩阵 (batch_size * max_length, hidden_size)
        acts1 = layer_activations_model1.view(-1, layer_activations_model1.shape[-1])
        acts2 = layer_activations_model2.view(-1, layer_activations_model2.shape[-1])

        # acts2_device = acts2.to(acts1.device)  # 将 acts2 移动到 acts1 所在的设备

        acts1_numpy = acts1.cpu().numpy()
        acts2_numpy = acts2.cpu().numpy()
        shape = "nd"

        # CCA
        calculate_cca(acts1_numpy, acts2_numpy, i, lang_sheet)

        # Alignment
        cal_Alignment(acts1_numpy, acts2_numpy, shape, i, lang_sheet)
        # RSM
        cal_RSM(acts1_numpy, acts2_numpy, shape, i, lang_sheet)
        # Neighbors
        cal_Neighbors(acts1_numpy, acts2_numpy, shape, i, lang_sheet)
        # Statistic
        cal_Statistic(acts1_numpy, acts2_numpy, shape, i, lang_sheet)


if __name__ == "__main__":

    device_model1 = torch.device("cuda:1")  # 第x块GPU
    device_model2 = torch.device("cuda:2")  # 第y块GPU

    device_model1 = 'cpu'
    device_model2 = 'cpu'

    # 模型和数据路径
    pt_model_7b = "/newdisk/public/wws/simMeasures/pt_file/CPP"
    pt_model_7b_Python = "/newdisk/public/wws/simMeasures/pt_file/CPP"
    
    # 调用主函数
    main(pt_model_7b, pt_model_7b_Python, device_model1, device_model2)
            


