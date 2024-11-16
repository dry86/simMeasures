import time 
import torch
import json5
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from example import cca_core
from utils import ResultSaver, combine_names
from functools import wraps
from getHiddenStates import concatenate_hidden_states, only_first_pt_hidden_states
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
    # saver.print_and_save("LinRegre", calculate_linear_regression(acts1, acts2, shape), idx)
    saver.print_and_save("AliCosSim", calculate_aligned_cosine_sim(acts1, acts2, shape), idx)
    saver.print_and_save("SoftCorMatch", calculate_soft_correlation_match(acts1, acts2, shape), idx)
    saver.print_and_save("HardCorMatch", calculate_hard_correlation_match(acts1, acts2, shape), idx)
    # saver.print_and_save("PermProc", calculate_permutation_procrustes(acts1, acts2, shape), idx)
    # saver.print_and_save("ProcDict", calculate_procrustes_size_and_shape_distance(acts1, acts2, shape), idx)

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



def main(task, num_layers_to_select, model1_path, model2_path, model_idx1, model_idx2, lang, device1, device2, saver: ResultSaver):
    """主函数：加载模型、读取数据、计算相似性"""

    # 获取隐藏层输出, shape (batch_size, max_length, hidden_size)
    pt_model_1 = model1_path + f"/pt_file/{task}/{lang}/"       # os.path.join(model1_path, "pt_file", task, lang)
    pt_model_2 = model2_path + f"/pt_file/{task}/{lang}/"   
    hidden_states_model1 = only_first_pt_hidden_states(pt_model_1, model_idx1, device1)
    hidden_states_model2 = only_first_pt_hidden_states(pt_model_2, model_idx2, device2)

    # 选择前num_layers_to_select层和后num_layers_to_select层
    selected_layers_1 = torch.cat((hidden_states_model1[:num_layers_to_select], hidden_states_model1[-num_layers_to_select:]), dim=0)
    selected_layers_2 = torch.cat((hidden_states_model2[:num_layers_to_select], hidden_states_model2[-num_layers_to_select:]), dim=0)
    
    for i in tqdm(range(2 * num_layers_to_select)):
        # 获取当前层的隐藏状态
        layer_hidden_states_1 = selected_layers_1[i]
        layer_hidden_states_2 = selected_layers_2[i]
        
        # 转换为numpy数组
        acts1_numpy = layer_hidden_states_1.cpu().numpy()
        acts2_numpy = layer_hidden_states_2.cpu().numpy()

        layer_idx = i
        # 标记后num_layers_to_select层从多少层开始（只在后半部分需要）
        if i >= num_layers_to_select:
            layer_idx = i + num_layers_to_select  # 后10层的索引

        shape = "nd"

        # CCA
        # calculate_cca(acts1_numpy, acts2_numpy, shape, layer_idx, saver)
        # Alignment
        cal_Alignment(acts1_numpy, acts2_numpy, shape, layer_idx, saver)
        # RSM
        cal_RSM(acts1_numpy, acts2_numpy, shape, layer_idx, saver)
        # Neighbors
        cal_Neighbors(acts1_numpy, acts2_numpy, shape, layer_idx, saver)
        # Topology
        # cal_Topology(acts1_numpy, acts2_numpy, shape, layer_idx, saver)
        # Statistic
        cal_Statistic(acts1_numpy, acts2_numpy, shape, layer_idx, saver)


if __name__ == "__main__":

    # 记录开始时间
    start_time = time.time()    

    device_model1 = torch.device("cuda:2")  # 第x块GPU
    device_model2 = torch.device("cuda:3")  # 第y块GPU

    # device_model1 = torch.device("cpu")  # 第x块GPU
    # device_model2 = torch.device("cpu")  # 第y块GPU

    # device_model1 = 'cpu'
    # device_model2 = 'cpu'

    # 参数设置
    configs = json5.load(open('/newdisk/public/wws/simMeasures/config/config-non-homogeneous-models.json5'))    # M

    for config in configs:
        task = config.get('task')
        model_idx1 = config.get('model_idx1')
        model_idx2 = config.get('model_idx2')
        lang = config.get('lang')
        print(task, model_idx1, model_idx2, lang)
    print("-"*50)

    for config in configs:
        task = config.get('task')
        prefix_model_path_idx1 = config.get('prefix_model_path_idx1')
        prefix_model_path_idx2 = config.get('prefix_model_path_idx2')
        model_idx1 = config.get('model_idx1')
        model_idx2 = config.get('model_idx2')
        lang = config.get('lang')
        num_layers_to_select = config.get('num_layers_to_select')

        model_pair = model_idx1 + "-" + model_idx2
        saver_name = model_pair + f"-{task}"
        sheet_name = combine_names(model_idx1, model_idx2, lang)
        saver = ResultSaver(file_name=f"/newdisk/public/wws/simMeasures/results/final_strategy_non_homogeneous_models/{model_pair}/{saver_name}.xlsx", sheet=sheet_name)

        # 调用主函数
        model_1 = prefix_model_path_idx1 + model_idx1
        model_2 = prefix_model_path_idx2 + model_idx2
        print(f"Current work: {task}, Model: {model_idx1}, {model_idx2}, lang: {lang}")
        main(task, num_layers_to_select, model_1, model_2, model_idx1, model_idx2, lang, device_model1, device_model2, saver)
        print(f"Finish work: {task}, Model: {model_idx1}, {model_idx2}, lang: {lang}")
        print("-"*50)
        print("-"*50)
        print("-"*50)


        # 记录结束时间
        end_time = time.time()
        # 计算并打印程序运行时间
        elapsed_time = (end_time - start_time) / 60
        print(f"Program runtime: {elapsed_time:.2f} mins")    


