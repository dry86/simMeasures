import torch
from getHiddenStates import concatenate_hidden_states
import numpy as np
from tqdm import tqdm
from repsim.measures import *


def cal_RSM(acts1, acts2, shape):

    gulp = Gulp()
    score = gulp(acts1, acts2, shape)
    print("\t Gulp: ", score)

    eigenspace_overlap = EigenspaceOverlapScore()
    score = eigenspace_overlap(acts1, acts2, shape)
    print("\t EigenspaceOverlapScore: ", score)

    dCor = DistanceCorrelation()
    score = dCor(acts1, acts2, shape)
    print("\t DistanceCorrelation: ", score)

    cka = CKA()
    score = cka(acts1, acts2, shape)
    print("\t CKA: ", score)

    rsa = RSA()
    score = rsa(acts1, acts2, shape)
    print("\t RSA: ", score)

    rsm_norm_diff = RSMNormDifference()
    score = rsm_norm_diff(acts1, acts2, shape)
    print("\t RSMNormDifference: ", score)

    

def main(model1_path, model2_path, device1, device2):

    # 获取隐藏层输出
    hidden_states_model1 = concatenate_hidden_states(model1_path, "hsm1", device1)
    hidden_states_model2 = concatenate_hidden_states(model2_path, "hsm2", device2)

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
        cal_RSM(acts1_np, acts2_np, shape)


if __name__ == "__main__":

    device_model1 = torch.device("cuda:0")  # 第x块GPU
    device_model2 = torch.device("cuda:1")  # 第y块GPU

    # 模型和数据路径
    pt_model_7b = "/newdisk/public/wws/simMeasures/pt_file/Python"
    pt_model_7b_Python = "/newdisk/public/wws/simMeasures/pt_file/Python"
    
    # 调用主函数
    main(pt_model_7b, pt_model_7b_Python, device_model1, device_model2)
            


