import torch
from getHiddenStates import concatenate_hidden_states, concatenate_last_layer_hidden_states
import numpy as np
from example import cca_core
from tqdm import tqdm
from utils import print_and_save
import jsonlines
from transformers import AutoTokenizer
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

def attention_mask_dataset_humaneval(model_idx, lang, device):
    padding_max_length = {  # python 90%: 262, cpp 90%: 275, java 90%: 292, javascript 90%: 259, go 90%: 168
    "Python": 262,
    "CPP": 275,
    "Java": 292,
    "JavaScript": 259,
    "GO": 168
    }   
    data_file_path = f"/newdisk/public/wws/Dataset/humaneval-x-main/data/{lang.lower()}/data/humaneval.jsonl"
    prefix_model_path = "/newdisk/public/wws/model_dir/codellama/"
    model_path = prefix_model_path + model_idx
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 使用 eos_token 作为 pad_token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # 使用EOS时,向右填充

    prompts = []

    # 读取数据文件
    with jsonlines.open(data_file_path) as reader:
        for obj in reader:
            task_id = obj.get('task_id')
            task_number = int(task_id.split('/')[-1])
            prompt = obj.get('prompt')
            # print(f"Task ID: {task_id}")
            prompts.append(prompt)

    inputs = tokenizer(prompts,
                        return_tensors='pt', 
                        padding='max_length', 
                        max_length=padding_max_length[lang], 
                        truncation=True
                       ).to(device)
    
    return inputs['attention_mask']

def main(model1_path, model2_path, lang, model_idx1, model_idx2, device1, device2):
    """主函数：加载模型、读取数据、计算相似性"""
    lang_sheet = lang # 拿到模型对比的数据集的语言, 在写入时作为sheet名称

    # 获取隐藏层输出, shape (batch_size, max_length, hidden_size)
    last_layer_hidden_states_model1 = concatenate_last_layer_hidden_states(model1_path, model_idx1, device1)
    last_layer_hidden_states_model2 = concatenate_last_layer_hidden_states(model2_path, model_idx2, device2)

    attention_mask_1 = attention_mask_dataset_humaneval(model_idx1, lang, device1)
    attention_mask_2 = attention_mask_dataset_humaneval(model_idx2, lang, device2)
    # 计算每个序列中最后一个有意义 token 的位置
    last_non_padding_index_1 = attention_mask_1.sum(dim=1) - 1 
    last_non_padding_index_2 = attention_mask_2.sum(dim=1) - 1 

    # 获取每个样本中最后一个有意义 token 对应的隐藏状态
    batch_size_1 = last_layer_hidden_states_model1.size(0)
    batch_size_2 = last_layer_hidden_states_model2.size(0)

    # 使用索引操作提取最后一个有意义的 token 的隐藏状态
    # acts1 = last_layer_hidden_states_model1[torch.arange(batch_size_1), last_non_padding_index_1, :]
    # acts2 = last_layer_hidden_states_model2[torch.arange(batch_size_2), last_non_padding_index_2, :]

    next_token_index_1 = torch.clamp(last_non_padding_index_1 - 2, max=262 - 1)
    next_token_index_2 = torch.clamp(last_non_padding_index_2 - 2, max=262 - 1)
    acts1 = last_layer_hidden_states_model1[torch.arange(batch_size_1), next_token_index_1, :]
    acts2 = last_layer_hidden_states_model2[torch.arange(batch_size_2), next_token_index_2, :]

    # acts1 = last_layer_hidden_states_model1[:, last_non_padding_index_1, :]   # 得到(164, 164, 4096)
    # acts2 = last_layer_hidden_states_model2[:, last_non_padding_index_2, :]
    # acts1 = last_layer_hidden_states_model1[:, -1, :]
    # acts2 = last_layer_hidden_states_model2[:, -1, :]

    acts1_numpy = acts1.cpu().numpy()
    acts2_numpy = acts2.cpu().numpy()
    shape = "nd"

    i = 10  
    # CCA
    # calculate_cca(acts1_numpy, acts2_numpy, i, lang_sheet)
    # Alignment
    cal_Alignment(acts1_numpy, acts2_numpy, shape, i, lang_sheet)
    # RSM
    cal_RSM(acts1_numpy, acts2_numpy, shape, i, lang_sheet)
    # Neighbors
    cal_Neighbors(acts1_numpy, acts2_numpy, shape, i, lang_sheet)
    # Statistic
    cal_Statistic(acts1_numpy, acts2_numpy, shape, i, lang_sheet)


if __name__ == "__main__":

    # 记录开始时间
    start_time = time.time()    

    device_model1 = torch.device("cuda:2")  # 第x块GPU
    device_model2 = torch.device("cuda:3")  # 第y块GPU

    # device_model1 = 'cpu'
    # device_model2 = 'cpu'

    # 参数设置

    prefix_pt_model = "/newdisk/public/wws/simMeasures/pt_file/"
    
    lang = "Python"
    model_idx1 = "codeLlama-7b"
    model_idx2 = "codeLlama-7b-Python"
    
    pt_model_1 = prefix_pt_model + lang + "/" + model_idx1
    pt_model_2 = prefix_pt_model + lang + "/" + model_idx2

    # 调用主函数
    main(pt_model_1, pt_model_2, lang, model_idx1, model_idx2, device_model1, device_model2)
    
    # print("Python, codeLlama7b Python, CCA epsilon=1e-8")


    # 记录结束时间
    end_time = time.time()
    # 计算并打印程序运行时间
    elapsed_time = end_time - start_time
    print(f"Program runtime: {elapsed_time:.2f} seconds")    


