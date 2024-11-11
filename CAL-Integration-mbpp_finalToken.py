import time 
import torch
import json5
import jsonlines
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from example import cca_core
from utils import ResultSaver
from functools import wraps
from getHiddenStates import concatenate_hidden_states, concatenate_last_layer_hidden_states
from repsim.measures.procrustes import orthogonal_procrustes
from repsim.measures.nearest_neighbor import joint_rank_jaccard_similarity
from repsim.measures import *

PRINT_TIMING = True # 通过设置此变量来控制是否打印运行时间

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if PRINT_TIMING:
            start_time = time.time()
        result = func(*args, **kwargs)
        if PRINT_TIMING:
            end_time = time.time()
            print(f"\t Time taken for {func.__name__}: {(end_time - start_time) / 60:.4f} mins")
        return result
    return wrapper

def cal_Statistic(acts1, acts2, shape, idx, saver):

    @time_it
    def calculate_magnitude_difference(acts1, acts2, shape):
        difference = MagnitudeDifference()
        return difference(acts1, acts2, shape)

    @time_it
    def calculate_concentricity_difference(acts1, acts2, shape):
        difference = ConcentricityDifference()
        return difference(acts1, acts2, shape)

    @time_it
    def calculate_uniformity_difference(acts1, acts2, shape):
        difference = UniformityDifference()
        return difference(acts1, acts2, shape)

    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")

    score = calculate_magnitude_difference(acts1, acts2, shape)
    saver.print_and_save("MagDiff", score, idx)

    score = calculate_concentricity_difference(acts1, acts2, shape)
    saver.print_and_save("ConDiff", score, idx)

    score = calculate_uniformity_difference(acts1, acts2, shape)
    saver.print_and_save("UniDiff", score, idx)

def cal_Topology(acts1, acts2, shape, idx, saver):

    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")

    @time_it
    def calculate_imd_score(acts1, acts2, shape):
        imd = IMDScore()
        return imd(acts1, acts2, shape)

    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")

    score = calculate_imd_score(acts1, acts2, shape)
    saver.print_and_save("IMD", score, idx)

def cal_Neighbors(acts1, acts2, shape, idx, saver):

    @time_it
    def calculate_jaccard_similarity(acts1, acts2, shape):
        jaccard = JaccardSimilarity()
        return jaccard(acts1, acts2, shape)

    @time_it
    def calculate_second_order_cosine_similarity(acts1, acts2, shape):
        secondOrder_cosine = SecondOrderCosineSimilarity()
        return secondOrder_cosine(acts1, acts2, shape)

    @time_it
    def calculate_rank_similarity(acts1, acts2, shape):
        rankSim = RankSimilarity()
        return rankSim(acts1, acts2, shape)

    @time_it
    def calculate_joint_rank_jaccard_similarity(acts1, acts2, shape):
        return joint_rank_jaccard_similarity(acts1, acts2, shape)

    # 主体逻辑
    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")

    score = calculate_jaccard_similarity(acts1, acts2, shape)
    saver.print_and_save("JacSim", score, idx)

    score = calculate_second_order_cosine_similarity(acts1, acts2, shape)
    saver.print_and_save("SecOrdCosSim", score, idx)

    score = calculate_rank_similarity(acts1, acts2, shape)
    saver.print_and_save("RankSim", score, idx)

    score = calculate_joint_rank_jaccard_similarity(acts1, acts2, shape)
    saver.print_and_save("RankJacSim", score, idx)

def cal_RSM(acts1, acts2, shape, idx, saver):
    @time_it
    def calculate_rsm_norm_difference(acts1, acts2, shape):
        rsm_norm_diff = RSMNormDifference()
        return rsm_norm_diff(acts1, acts2, shape)

    @time_it
    def calculate_rsa(acts1, acts2, shape):
        rsa = RSA()
        return rsa(acts1, acts2, shape)

    @time_it
    def calculate_cka(acts1, acts2, shape):
        cka = CKA()
        return cka(acts1, acts2, shape)

    @time_it
    def calculate_distance_correlation(acts1, acts2, shape):
        dCor = DistanceCorrelation()
        return dCor(acts1, acts2, shape)

    @time_it
    def calculate_eigenspace_overlap(acts1, acts2, shape):
        eigenspace_overlap = EigenspaceOverlapScore()
        return eigenspace_overlap(acts1, acts2, shape)

    @time_it
    def calculate_gulp(acts1, acts2, shape):
        gulp = Gulp()
        return gulp(acts1, acts2, shape)

    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")

    saver.print_and_save("RSMNormDiff", calculate_rsm_norm_difference(acts1, acts2, shape), idx)
    saver.print_and_save("RSA", calculate_rsa(acts1, acts2, shape), idx)
    saver.print_and_save("CKA", calculate_cka(acts1, acts2, shape), idx)
    saver.print_and_save("DisCor", calculate_distance_correlation(acts1, acts2, shape), idx)
    saver.print_and_save("EOlapScore", calculate_eigenspace_overlap(acts1, acts2, shape), idx)
    # assert K <= n  -> AssertionError
    # saver.print_and_save("Gulp", calculate_gulp(acts1, acts2, shape), idx)

def cal_Alignment(acts1, acts2, shape, idx, saver):

    @time_it
    def calculate_opcan(acts1, acts2, shape):
        opcan = OrthogonalProcrustesCenteredAndNormalized()
        return opcan(acts1, acts2, shape)

    @time_it
    def calculate_asm(acts1, acts2, shape):
        asm = OrthogonalAngularShapeMetricCentered()
        return asm(acts1, acts2, shape)

    @time_it
    def calculate_linear_regression(acts1, acts2, shape):
        linear_regression = LinearRegression()
        return linear_regression(acts1, acts2, shape)

    @time_it
    def calculate_aligned_cosine_sim(acts1, acts2, shape):
        aligned_cosine_sim = AlignedCosineSimilarity()
        return aligned_cosine_sim(acts1, acts2, shape)

    @time_it
    def calculate_soft_correlation_match(acts1, acts2, shape):
        soft_correlation_match = SoftCorrelationMatch()
        return soft_correlation_match(acts1, acts2, shape)

    @time_it
    def calculate_hard_correlation_match(acts1, acts2, shape):
        hard_correlation_match = HardCorrelationMatch()
        return hard_correlation_match(acts1, acts2, shape)

    @time_it
    def calculate_permutation_procrustes(acts1, acts2, shape):
        PermProc = PermutationProcrustes()
        return PermProc(acts1, acts2, shape)

    @time_it
    def calculate_procrustes_size_and_shape_distance(acts1, acts2, shape):
        ProcDict = ProcrustesSizeAndShapeDistance()
        return ProcDict(acts1, acts2, shape)

    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")

    saver.print_and_save("OrthProCAN", calculate_opcan(acts1, acts2, shape), idx)
    saver.print_and_save("OrthAngShape", calculate_asm(acts1, acts2, shape), idx)
    saver.print_and_save("LinRegre", calculate_linear_regression(acts1, acts2, shape), idx)
    saver.print_and_save("AliCosSim", calculate_aligned_cosine_sim(acts1, acts2, shape), idx)
    saver.print_and_save("SoftCorMatch", calculate_soft_correlation_match(acts1, acts2, shape), idx)
    saver.print_and_save("HardCorMatch", calculate_hard_correlation_match(acts1, acts2, shape), idx)
    saver.print_and_save("PermProc", calculate_permutation_procrustes(acts1, acts2, shape), idx)
    saver.print_and_save("ProcDict", calculate_procrustes_size_and_shape_distance(acts1, acts2, shape), idx)

def calculate_cca(acts1, acts2, shape, idx, saver):
    @time_it
    def calculate_cca(acts1, acts2):
        results = cca_core.get_cca_similarity(acts1, acts2, epsilon=1e-6, verbose=False)
        return results

    @time_it
    def calculate_svcca(acts1, acts2, shape):
        svcca = SVCCA()
        return svcca(acts1.T, acts2.T, shape)

    @time_it
    def calculate_pwcca(results, acts1, acts2):
        pwcca_mean, _, _ = cca_core.compute_pwcca(results, acts1, acts2)
        return pwcca_mean

    acts1 = acts1.T # convert to neurons by datapoints
    acts2 = acts2.T
    print(f"Layer {idx}, acts1 shape: {acts1.shape}:")

    results = calculate_cca(acts1, acts2)
    saver.print_and_save("MeanCCA", np.mean(results["cca_coef1"]), row=idx)
    # svcca_res = cca_core.compute_svcca(acts1, acts2)
    saver.print_and_save("SVCCA", calculate_svcca(acts1, acts2, shape), row=idx)    # SVCCA transpose acts
    saver.print_and_save("PWCCA", calculate_pwcca(results, acts1, acts2), row=idx)



def main(model1_path, model2_path, model_idx1, model_idx2, lang, device1, device2, saver: ResultSaver):
    """主函数：加载模型、读取数据、计算相似性"""

    # 获取隐藏层输出, shape (batch_size, max_length, hidden_size)
    pt_model_1 = model1_path + f"/pt_file/textGen_MBPP/{lang}/"   # M
    pt_model_2 = model2_path + f"/pt_file/textGen_MBPP/{lang}/"
    hidden_states_model1 = concatenate_hidden_states(pt_model_1, model_idx1, device1)
    hidden_states_model2 = concatenate_hidden_states(pt_model_2, model_idx2, device2)

    # 获取模型的总层数并计算每一层的 score
    num_layers = len(hidden_states_model1)
    for i in tqdm(range(num_layers)):

        # if i < 13:
        #     continue

        layer_hidden_states_1 = hidden_states_model1[i]
        layer_hidden_states_2 = hidden_states_model2[i]

        acts1_numpy = layer_hidden_states_1.cpu().numpy()
        acts2_numpy = layer_hidden_states_2.cpu().numpy()
        shape = "nd"

        # CCA
        # calculate_cca(acts1_numpy, acts2_numpy, shape, i, saver)
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

    device_model1 = torch.device("cuda:0")  # 第x块GPU
    device_model2 = torch.device("cuda:1")  # 第y块GPU

    # device_model1 = 'cpu'
    # device_model2 = 'cpu'

    # 参数设置
    configs = json5.load(open('/newdisk/public/wws/simMeasures/config/config-mbpp.json5'))  # M

    for config in configs:
        prefix_model_path = config.get('prefix_model_path')
        model_idx1 = config.get('model_idx1')
        model_idx2 = config.get('model_idx2')
        lang = config.get('lang')
        print(prefix_model_path, model_idx1, model_idx2, lang)
    print("-"*50)

    for config in configs:
        prefix_model_path = config.get('prefix_model_path')
        model_idx1 = config.get('model_idx1')
        model_idx2 = config.get('model_idx2')
        lang = config.get('lang')

        model_pair = model_idx1 + "-" + model_idx2
        saver_name = model_pair + "-mbpp_finalToken" # M
        sheet_name = model_idx1 + "-" + model_idx2.split("-")[-1] + "-" + lang
        saver = ResultSaver(file_name=f"/newdisk/public/wws/simMeasures/results/final_strategy/{model_pair}/{saver_name}.xlsx", sheet=sheet_name)

        # 调用主函数
        model_1 = prefix_model_path + model_idx1
        model_2 = prefix_model_path + model_idx2
        print(f"Current work: {model_pair}, lang: {lang}, CCA series epsilon=1e-6")
        main(model_1, model_2, model_idx1, model_idx2, lang, device_model1, device_model2, saver)
        print(f"Finish work: {model_pair}, lang: {lang}, CCA series epsilon=1e-6")
        print("-"*50)
        print("-"*50)
        print("-"*50)


        # 记录结束时间
        end_time = time.time()
        # 计算并打印程序运行时间
        elapsed_time = (end_time - start_time) / 60
        print(f"Program runtime: {elapsed_time:.2f} mins")    


