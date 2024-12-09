import os
import sys
import time 
import torch
import json5
import numpy as np
sys.path.append("/newdisk/public/wws/simMeasures")
from tqdm import tqdm
from utils import ResultSaver
from functools import wraps
from getHiddenStates import concatenate_hidden_states, only_first_pt_hidden_states
from repsim.measures.nearest_neighbor import joint_rank_jaccard_similarity
from repsim.measures import *
from itertools import combinations

PRINT_TIMING = False # 通过设置此变量来控制是否打印运行时间

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

def cal_RSM(acts1, acts2, shape, idx, lang_pair, saver):
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

    # saver.print_and_save("RSMNormDiff", calculate_rsm_norm_difference(acts1, acts2, shape), idx)
    # saver.print_and_save("RSA", calculate_rsa(acts1, acts2, shape), idx)
    saver.print_and_save(f"{lang_pair}_CKA", calculate_cka(acts1, acts2, shape), idx)
    # saver.print_and_save("DisCor", calculate_distance_correlation(acts1, acts2, shape), idx)
    # saver.print_and_save("EOlapScore", calculate_eigenspace_overlap(acts1, acts2, shape), idx)
    # assert K <= n  -> AssertionError
    # saver.print_and_save("Gulp", calculate_gulp(acts1, acts2, shape), idx)


def main(task, num_layers_to_select, model1_path, model_idx1, langs, device, saver: ResultSaver):
    """主函数：加载模型、读取数据、计算相似性"""

    # 用于保存语言对应的隐藏状态
    lang_hidden_states = {}

    # 遍历语言列表，加载每种语言的隐藏状态
    for lang in langs:
        pt_model_path = os.path.join(model1_path, "pt_file", task, lang)
        hidden_states = only_first_pt_hidden_states(pt_model_path, model_idx1, device)
        lang_hidden_states[lang] = hidden_states

    # 两两语言比较
    for lang1, lang2 in combinations(langs, 2):
        print(f"Comparing {lang1} and {lang2}...")
        lang_pair = f"{lang1}-{lang2}"

        hidden_states_1 = lang_hidden_states[lang1]
        hidden_states_2 = lang_hidden_states[lang2]

        # 选择前 num_layers_to_select 层和后 num_layers_to_select 层
        selected_layers_1 = torch.cat((hidden_states_1[:num_layers_to_select], hidden_states_1[-num_layers_to_select:]), dim=0)
        selected_layers_2 = torch.cat((hidden_states_2[:num_layers_to_select], hidden_states_2[-num_layers_to_select:]), dim=0)
    
        for i in tqdm(range(2 * num_layers_to_select)):
            # 获取当前层的隐藏状态
            layer_hidden_states_1 = selected_layers_1[i]
            layer_hidden_states_2 = selected_layers_2[i]
            
            # 转换为numpy数组
            acts1_numpy = layer_hidden_states_1.cpu().numpy()
            acts2_numpy = layer_hidden_states_2.cpu().numpy()

            # 在调用 CKA 之前，确保 numpy 数组维度匹配
            min_len = min(acts1_numpy.shape[0], acts2_numpy.shape[0])
            acts1_numpy = acts1_numpy[:min_len, :]
            acts2_numpy = acts2_numpy[:min_len, :]

            layer_idx = i
            # 标记后num_layers_to_select层从多少层开始（只在后半部分需要）
            if i >= num_layers_to_select:
                layer_idx = i + num_layers_to_select  # 后10层的索引

            shape = "nd"

            # RSM
            cal_RSM(acts1_numpy, acts2_numpy, shape, layer_idx, lang_pair, saver)



if __name__ == "__main__":

    # 记录开始时间
    start_time = time.time()    

    device_model1 = torch.device("cuda:2")  # 第x块GPU

    # 参数设置
    configs = json5.load(open(
        '/newdisk/public/wws/simMeasures/config/config-CKA-PLs.json5'))    # M

    for config in configs:
        task = config.get('task')
        langs = config.get('lang_idx1')
        model = config.get('model_path')
        print(task, langs, model)
    print("-"*50)

    for config in configs:
        tasks = config.get('task')
        langs= config.get('langs')
        model_path = config.get('model_path')
        num_layers_to_select = config.get('num_layers_to_select')

        for task in tasks:
            model_idx1 = os.path.basename(model_path)

            saver_name = model_idx1
            sheet_name = task

            saver = ResultSaver(
                file_name=f"//newdisk/public/wws/simMeasures/pyplot/Hotmap/Languages/xlsx/{saver_name}.xlsx", 
                sheet=sheet_name)

            # 调用主函数
            print(f"Current work: {task}, Model: {model_idx1}, lang: {langs}")
            main(task, num_layers_to_select, model_path, model_idx1, langs, device_model1, saver)
            print(f"Finish work: {task}, Model: {model_idx1}, lang: {langs}")
            print("-"*50)
            print("-"*50)
            print("-"*50)


        # 记录结束时间
        end_time = time.time()
        # 计算并打印程序运行时间
        elapsed_time = (end_time - start_time) / 60
        print(f"Program runtime: {elapsed_time:.2f} mins")    


