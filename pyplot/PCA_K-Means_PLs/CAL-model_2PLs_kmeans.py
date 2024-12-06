import os
import sys 
import time 
import torch
import json5
from tqdm import tqdm
sys.path.append("/newdisk/public/wws/simMeasures")
from getHiddenStates import concatenate_hidden_states, only_first_pt_hidden_states

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def main(task, num_layers_to_select, model1_path, model_idx1, lang1, lang2, device1, device2):
    """主函数：加载模型、读取数据、计算相似性"""

    # 获取隐藏层输出, shape (batch_size, max_length, hidden_size)
    pt_model_1 = os.path.join(model1_path, "pt_file", task, lang1)   # 
    pt_model_2 = os.path.join(model1_path, "pt_file", task, lang2)
    hidden_states_model1 = only_first_pt_hidden_states(pt_model_1, model_idx1, device1)
    hidden_states_model2 = only_first_pt_hidden_states(pt_model_2, model_idx1, device2)

    python_hidden_states1 = hidden_states_model1[-1].cpu().numpy()
    other_hidden_states2 = hidden_states_model2[-1].cpu().numpy()
    N_python = python_hidden_states1.shape[0]
    N_java = other_hidden_states2.shape[0]

    X_python = python_hidden_states1
    X_java = other_hidden_states2
    # 假设最终 X_python: shape (N_python, d_model)
    #           X_java:   shape (N_java, d_model)
    X = np.concatenate([X_python, X_java], axis=0)
    labels = np.array(['Python'] * N_python + ['Java'] * N_java)

    # 降维到2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # # K-Means聚类
    # kmeans = KMeans(n_clusters=2, random_state=42)
    # cluster_assignments = kmeans.fit_predict(X_2d)

    # 可视化
    colors = ['blue' if l == 'Python' else 'red' for l in labels]

    plt.figure(figsize=(8,6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.7)
    plt.title('Python vs Java Embedding Space')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.savefig(f"simMeasures/pyplot/PCA_K-Means/1pca_{model_idx1}_{task}_{lang2}_{lang1}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    # 记录开始时间
    start_time = time.time()    

    device_model1 = torch.device("cuda:2")  # 第x块GPU
    device_model2 = torch.device("cuda:1")  # 第y块GPU

    # device_model1 = torch.device("cpu")  # 第x块GPU
    # device_model2 = torch.device("cpu")  # 第y块GPU

    # 参数设置
    configs = json5.load(open(
        '/newdisk/public/wws/simMeasures/config/config-PCA_KMeans.json5'))    # M

    for config in configs:
        task = config.get('task')
        lang_idx1_list = config.get('lang_idx1')
        lang_idx2 = config.get('lang_idx2')
        model = config.get('model_path')
        print(task, lang_idx2, lang_idx1_list, model)
    print("-"*50)

    for config in configs:
        tasks = config.get('task')
        lang_idx1_list = config.get('lang_idx1')
        lang_idx2 = config.get('lang_idx2')
        model_path = config.get('model_path')
        num_layers_to_select = config.get('num_layers_to_select')

        for task in tasks:
            for lang_idx1 in lang_idx1_list:
                model_idx1 = os.path.basename(model_path)

                # 调用主函数
                print(f"Current work: {task}, Model: {model_idx1}, lang: {lang_idx1}, {lang_idx2}")
                main(task, num_layers_to_select, model_path, model_idx1, lang_idx1, lang_idx2, device_model1, device_model2)
                print(f"Finish work: {task}, Model: {model_idx1}, lang: {lang_idx1}, {lang_idx2}")
                print("-"*50)
                print("-"*50)
                print("-"*50)

        # 记录结束时间
        end_time = time.time()
        # 计算并打印程序运行时间
        elapsed_time = (end_time - start_time) / 60
        print(f"Program runtime: {elapsed_time:.2f} mins")    

