import torch
from getHiddenStates import concatenate_hidden_states, concatenate_final_token_hidden_states
import numpy as np
from example import cca_core
from tqdm import tqdm
from utils import print_and_save

from repsim.measures.procrustes import orthogonal_procrustes
from repsim.measures.nearest_neighbor import joint_rank_jaccard_similarity
from repsim.measures import *
import time  # 导入 time 模块

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

    # assert K <= n  AssertionError
    # gulp = Gulp()
    # score = gulp(acts1, acts2, shape)
    # print_and_save("Gulp", score, idx, sheet)

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


def main(model1_path, model2_path, lang, model_idx1, model_idx2, device1, device2):
    """主函数：加载模型、读取数据、计算相似性"""
    lang_sheet = lang # 拿到模型对比的数据集的语言, 在写入时作为sheet名称

    # 获取隐藏层输出, shape (batch_size, max_length, hidden_size)
    last_layer_hidden_states_model1 = concatenate_final_token_hidden_states(model1_path, model_idx1, device1)
    last_layer_hidden_states_model2 = concatenate_final_token_hidden_states(model2_path, model_idx2, device2)

    # 通过 [:, -1, :] 只取(batch_size, max_length, hidden_size)中的最后一个token -> acts = (batch_size, hidden_size)
    acts1 = last_layer_hidden_states_model1[:, -1, :]
    acts2 = last_layer_hidden_states_model2[:, -1, :]

    # acts2_device = acts2.to(acts1.device)  # 将 acts2 移动到 acts1 所在的设备

    acts1_numpy = acts1.cpu().numpy()
    acts2_numpy = acts2.cpu().numpy()
    shape = "nd"

    i = 0
    # CCA
    # calculate_cca(acts1_numpy, acts2_numpy, i, lang_sheet)
    # Alignment
    # cal_Alignment(acts1_numpy, acts2_numpy, shape, i, lang_sheet)
    # # RSM
    # cal_RSM(acts1_numpy, acts2_numpy, shape, i, lang_sheet)
    # Neighbors
    cal_Neighbors(acts1_numpy, acts2_numpy, shape, i, lang_sheet)
    # Statistic
    cal_Statistic(acts1_numpy, acts2_numpy, shape, i, lang_sheet)


if __name__ == "__main__":

    # 记录开始时间
    start_time = time.time()    

    device_model1 = torch.device("cuda:0")  # 第x块GPU
    device_model2 = torch.device("cuda:1")  # 第y块GPU

    # device_model1 = 'cpu'
    # device_model2 = 'cpu'

    # 参数设置

    prefix_pt_model = "/newdisk/public/wws/simMeasures/pt_file/"
    
    lang = "Python"
    model_idx1 = "codeLlama7b"
    model_idx2 = "codeLlama7bPython"
    
    pt_model_1 = prefix_pt_model + lang + "/" + model_idx1
    pt_model_2 = prefix_pt_model + lang + "/" + model_idx2

    # 调用主函数
    main(pt_model_1, pt_model_2, lang, model_idx1, model_idx2, device_model1, device_model2)
    
    print("Python, codeLlama7b Python, CCA epsilon=1e-8")


    # 记录结束时间
    end_time = time.time()
    # 计算并打印程序运行时间
    elapsed_time = end_time - start_time
    print(f"Program runtime: {elapsed_time:.2f} seconds")    


