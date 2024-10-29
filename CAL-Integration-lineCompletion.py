import time 
import torch
import json5
import jsonlines
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from example import cca_core
from utils import ResultSaver
from getHiddenStates import concatenate_hidden_states, concatenate_last_layer_hidden_states
from repsim.measures.procrustes import orthogonal_procrustes
from repsim.measures.nearest_neighbor import joint_rank_jaccard_similarity
from repsim.measures import *

def cal_Statistic(acts1, acts2, shape, idx, saver):

    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")

    difference = MagnitudeDifference()
    score = difference(acts1, acts2, shape)
    saver.print_and_save("MagDiff", score, idx)

    difference = ConcentricityDifference()
    score = difference(acts1, acts2, shape)
    saver.print_and_save("ConDiff", score, idx)

    difference = UniformityDifference()
    score = difference(acts1, acts2, shape)
    saver.print_and_save("UniDiff", score, idx)

def cal_Topology(acts1, acts2, shape, idx, saver):

    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")

    imd = IMDScore()
    score = imd(acts1, acts2, shape)
    saver.print_and_save("IMD", score, idx)

def cal_Neighbors(acts1, acts2, shape, idx, saver):

    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")

    jaccard = JaccardSimilarity()
    score = jaccard(acts1, acts2, shape)
    saver.print_and_save("JacSim", score, idx)

    secondOrder_cosine = SecondOrderCosineSimilarity()
    score = secondOrder_cosine(acts1, acts2, shape)
    saver.print_and_save("SecOrdCosSim", score, idx)

    rankSim = RankSimilarity()
    score = rankSim(acts1, acts2, shape)
    saver.print_and_save("RankSim", score, idx)

    score = joint_rank_jaccard_similarity(acts1, acts2, shape)
    saver.print_and_save("RankJacSim", score, idx)

def cal_RSM(acts1, acts2, shape, idx, saver):

    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")

    # 计算耗时太长 !
    rsm_norm_diff = RSMNormDifference()
    score = rsm_norm_diff(acts1, acts2, shape)
    saver.print_and_save("RSMNormDiff", score, idx)

    rsa = RSA()
    score = rsa(acts1, acts2, shape)
    saver.print_and_save("RSA", score, idx)

    cka = CKA()
    score = cka(acts1, acts2, shape)
    saver.print_and_save("CKA", score, idx)

    dCor = DistanceCorrelation()
    score = dCor(acts1, acts2, shape)
    saver.print_and_save("DisCor", score, idx)

    eigenspace_overlap = EigenspaceOverlapScore()
    score = eigenspace_overlap(acts1, acts2, shape)
    saver.print_and_save("EOlapScore", score, idx)

    # assert K <= n  -> AssertionError
    gulp = Gulp()
    score = gulp(acts1, acts2, shape)
    saver.print_and_save("Gulp", score, idx)

def cal_Alignment(acts1, acts2, shape, idx, saver):

    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")

    # score = orthogonal_procrustes(acts1, acts2, shape)
    # saver.print_and_save("OrthPro", score, idx)

    opcan = OrthogonalProcrustesCenteredAndNormalized()
    score = opcan(acts1, acts2, shape)
    saver.print_and_save("OrthProCAN", score, idx)

    asm = OrthogonalAngularShapeMetricCentered()
    score = asm(acts1, acts2, shape)
    saver.print_and_save("OrthAngShape", score, idx)

    linear_regression = LinearRegression()
    score = linear_regression(acts1, acts2, shape)
    saver.print_and_save("LinRegre", score, idx)

    aligned_cosine_sim = AlignedCosineSimilarity()
    score = aligned_cosine_sim(acts1, acts2, shape)
    saver.print_and_save("AliCosSim", score, idx)

    soft_correlation_match = SoftCorrelationMatch()
    score = soft_correlation_match(acts1, acts2, shape)
    saver.print_and_save("SoftCorMatch", score, idx)

    hard_correlation_match = HardCorrelationMatch()
    score = hard_correlation_match(acts1, acts2, shape)
    saver.print_and_save("HardCorMatch", score, idx)

    PermProc = PermutationProcrustes()
    score = PermProc(acts1, acts2, shape)
    saver.print_and_save("PermProc", score, idx)

    ProcDict = ProcrustesSizeAndShapeDistance()
    score = ProcDict(acts1, acts2, shape)
    saver.print_and_save("ProcDict", score, idx)

def calculate_cca(acts1, acts2, idx, saver):

    acts1 = acts1.T # convert to neurons by datapoints
    acts2 = acts2.T
    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")
    results = cca_core.get_cca_similarity(acts1, acts2, epsilon=1e-6, verbose=False)
    saver.print_and_save("MeanCCA", np.mean(results["cca_coef1"]), row=idx)

    svcca_res = cca_core.compute_svcca(acts1, acts2)
    saver.print_and_save("SVCCA", svcca_res, row=idx)

    pwcca_mean, w, _ = cca_core.compute_pwcca(results, acts1, acts2)
    saver.print_and_save("PWCCA", pwcca_mean, row=idx)


def main(model1_path, model2_path, model_idx1, model_idx2, lang, device1, device2, saver: ResultSaver):
    """主函数：加载模型、读取数据、计算相似性"""

    # 获取隐藏层输出, shape (batch_size, max_length, hidden_size)
    pt_model_1 = model1_path + f"/pt_file/line_completion/{lang}/"
    pt_model_2 = model2_path + f"/pt_file/line_completion/{lang}/"
    hidden_states_model1 = concatenate_hidden_states(pt_model_1, model_idx1, device1)
    hidden_states_model2 = concatenate_hidden_states(pt_model_2, model_idx2, device2)

    # 获取模型的总层数并计算每一层的 score
    num_layers = len(hidden_states_model1)
    for i in tqdm(range(num_layers)):

        layer_hidden_states_1 = hidden_states_model1[i]
        layer_hidden_states_2 = hidden_states_model2[i]

        acts1_numpy = layer_hidden_states_1.cpu().numpy()
        acts2_numpy = layer_hidden_states_2.cpu().numpy()
        shape = "nd"

        # CCA
        # calculate_cca(acts1_numpy, acts2_numpy, i, saver)
        # Alignment
        cal_Alignment(acts1_numpy, acts2_numpy, shape, i, saver)
        # RSM
        cal_RSM(acts1_numpy, acts2_numpy, shape, i, saver)
        # Neighbors
        cal_Neighbors(acts1_numpy, acts2_numpy, shape, i, saver)
        # Topology
        cal_Topology(acts1_numpy, acts2_numpy, shape, i, saver)
        # Statistic
        cal_Statistic(acts1_numpy, acts2_numpy, shape, i, saver)


if __name__ == "__main__":

    # 记录开始时间
    start_time = time.time()    

    device_model1 = torch.device("cuda:2")  # 第x块GPU
    device_model2 = torch.device("cuda:3")  # 第y块GPU

    # device_model1 = 'cpu'
    # device_model2 = 'cpu'

    # 参数设置
    configs = json5.load(open('/newdisk/public/wws/simMeasures/config/config-lineCompletion.json5'))

    for config in configs:
        prefix_model_path = config.get('prefix_model_path')
        model_idx1 = config.get('model_idx1')
        model_idx2 = config.get('model_idx2')
        lang = config.get('lang')

        model_pair = model_idx1 + "-" + model_idx2
        saver_name = model_pair + "-lineCompletion"
        sheet_name = model_idx1 + "-" + model_idx2.split("-")[-1] + "-" + lang
        saver = ResultSaver(file_name=f"/newdisk/public/wws/simMeasures/results/final_strategy/{model_pair}/{saver_name}.xlsx", sheet=sheet_name)

        # 调用主函数
        model_1 = prefix_model_path + model_idx1
        model_2 = prefix_model_path + model_idx2
        print(f"Current work: {model_pair}, lang: {lang}, CCA epsilon=1e-6")
        main(model_1, model_2, model_idx1, model_idx2, lang, device_model1, device_model2, saver)
        print(f"Finish work: {model_pair}, lang: {lang}, CCA epsilon=1e-6")
        print("-"*50)
        print("-"*50)
        print("-"*50)


        # 记录结束时间
        end_time = time.time()
        # 计算并打印程序运行时间
        elapsed_time = (end_time - start_time) / 60
        print(f"Program runtime: {elapsed_time:.2f} mins")    


