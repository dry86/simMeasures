import torch
from getHiddenStates import load_hidden_states
import numpy as np
from tqdm import tqdm
from repsim.measures import *
from repsim.measures.nearest_neighbor import joint_rank_jaccard_similarity

def cal_Neighbors(acts1, acts2, shape):

    jaccard = JaccardSimilarity()
    score = jaccard(acts1, acts2, shape)
    print("\t JaccardSimilarity: ", score)

    secondOrder_cosine = SecondOrderCosineSimilarity()
    score = secondOrder_cosine(acts1, acts2, shape)
    print("\t SecondOrderCosineSimilarity: ", score)

    rankSim = RankSimilarity()
    score = rankSim(acts1, acts2, shape)
    print("\t RankSimilarity: ", score)


    score = joint_rank_jaccard_similarity(acts1, acts2, shape)
    print("\t joint_rank_jaccard_similarity: ", score)


    

def main(model1_path, model2_path, device1, device2):

    # 获取隐藏层输出
    hidden_states_model1 = load_hidden_states(model1_path, device1)
    hidden_states_model2 = load_hidden_states(model2_path, device2)

    # 获取模型的总层数并计算每一层的 相关性得分
    num_layers = len(hidden_states_model1)
    for i in tqdm(range(num_layers)):

        # 先将每层所有数据的隐藏层激活拼接成三维矩阵 (batch_size, max_length, hidden_size)
        layer_activations_model1 = hidden_states_model1[i]  # 形状 (batch_size, max_length, hidden_size)
        layer_activations_model2 = hidden_states_model2[i]  # 形状 (batch_size, max_length, hidden_size)

        # 通过 view() 函数将其变成二维矩阵 (batch_size * max_length, hidden_size)
        acts1 = layer_activations_model1.view(-1, layer_activations_model1.shape[-1])
        acts2 = layer_activations_model2.view(-1, layer_activations_model2.shape[-1])
        print(f"Layer {i}, acts1 shape: {acts1.shape}:")

        acts1_np = acts1.cpu().numpy()
        acts2_np = acts2.cpu().numpy()

        shape = "nd"
        # 计算
        cal_Neighbors(acts1_np, acts2_np, shape)


if __name__ == "__main__":

    device_model1 = torch.device("cuda:0")  # 第x块GPU
    device_model2 = torch.device("cuda:1")  # 第y块GPU

    device_model1 = 'cpu'
    device_model2 = 'cpu'

    # 模型和数据路径
    pt_model_7b = "/newdisk/public/wws/simMeasures/pt_file/Python/hsm1_batch_19.pt"
    pt_model_7b_Python = "/newdisk/public/wws/simMeasures/pt_file/Python/hsm2_batch_19.pt"
    
    # 调用主函数
    main(pt_model_7b, pt_model_7b_Python, device_model1, device_model2)
            


