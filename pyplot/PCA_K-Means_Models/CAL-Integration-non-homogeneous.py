import os
import sys
import time 
import torch
import json5

sys.path.append("/newdisk/public/wws/simMeasures")
from getHiddenStates import concatenate_hidden_states, only_first_pt_hidden_states

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def main(task, num_layers_to_select, model1_path, model2_path, model_idx1, model_idx2, lang, device1, device2):
    """主函数：加载模型、读取数据、计算相似性"""

    # 获取隐藏层输出, shape (batch_size, max_length, hidden_size)
    pt_model_1 = os.path.join(model1_path, "pt_file", task, lang)   # 
    pt_model_2 = os.path.join(model2_path, "pt_file", task, lang)
    hidden_states_model1 = only_first_pt_hidden_states(pt_model_1, model_idx1, device1)
    hidden_states_model2 = only_first_pt_hidden_states(pt_model_2, model_idx2, device2)

    last_layer_hidden_states1 = hidden_states_model1[0].cpu().numpy()
    last_layer_hidden_states2 = hidden_states_model2[0].cpu().numpy()

    N_model1 = last_layer_hidden_states1.shape[0]
    N_model2 = last_layer_hidden_states2.shape[0]

    X_model1 = last_layer_hidden_states1
    X_model2 = last_layer_hidden_states2
    # 假设最终 X_model1: shape (N_python, d_model)
    #           X_model2:   shape (N_java, d_model)
    X = np.concatenate([X_model1, X_model2], axis=0)
    labels = np.array(['Model_1'] * N_model1 + ['Model_2'] * N_model2)

    # 降维到2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # # K-Means聚类
    # kmeans = KMeans(n_clusters=2, random_state=42)
    # cluster_assignments = kmeans.fit_predict(X_2d)

    # 可视化
    colors = ['blue' if l == 'Model_1' else 'red' for l in labels]

    plt.figure(figsize=(8,6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.7)
    plt.title('Model_1 vs Model_2 Embedding Space')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.savefig(f"simMeasures/pyplot/PCA_K-Means_Models/0Layer_PCA_{model_idx1}_{model_idx2}_{task}_{lang}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()




if __name__ == "__main__":

    # 记录开始时间
    start_time = time.time()    

    device_model1 = torch.device("cuda:2")  # 第x块GPU
    device_model2 = torch.device("cuda:1")  # 第y块GPU


    # 参数设置
    configs = json5.load(open(
        '/newdisk/public/wws/simMeasures/config/config-PCA_KMeans-Models.json5'))    # M

    for config in configs:
        task = config.get('task')
        prefix_model_path_idx1_list = config.get('prefix_model_path_idx1')
        prefix_model_path_idx2 = config.get('prefix_model_path_idx2')
        lang = config.get('lang')
        print(task, prefix_model_path_idx2, prefix_model_path_idx1_list, lang)
    print("-"*50)

    for config in configs:
        tasks = config.get('task')
        prefix_model_path_idx1_list = config.get('prefix_model_path_idx1')
        prefix_model_path_idx2 = config.get('prefix_model_path_idx2')
        lang = config.get('lang')
        num_layers_to_select = config.get('num_layers_to_select')

        for prefix_model_path_idx1 in prefix_model_path_idx1_list:
            for task in tasks:
                model_idx1 = os.path.basename(prefix_model_path_idx1)
                model_idx2 = os.path.basename(prefix_model_path_idx2)

                # 调用主函数
                print(f"Current work: {task}, Model: {model_idx2}, {model_idx1}, lang: {lang}")
                main(task, num_layers_to_select, prefix_model_path_idx1, prefix_model_path_idx2, model_idx1, model_idx2, lang, device_model1, device_model2)
                print(f"Finish work: {task}, Model: {model_idx2}, {model_idx1}, lang: {lang}")
                print("-"*50)
                print("-"*50)
                print("-"*50)

        # 记录结束时间
        end_time = time.time()
        # 计算并打印程序运行时间
        elapsed_time = (end_time - start_time) / 60
        print(f"Program runtime: {elapsed_time:.2f} mins")    


