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
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

# 函数：计算协方差矩阵的椭圆参数
def get_ellipse_params(points):
    cov = np.cov(points, rowvar=False)  # 计算协方差矩阵
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # 特征值和特征向量
    # 按特征值大小排序
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))  # 计算旋转角度
    width, height = 2 * np.sqrt(eigenvalues)  # 椭圆轴长为特征值的平方根的2倍
    return width, height, angle


def main(task, num_layers_to_select, model1_path, model_idx1, langs, device):

    hidden_states = []
    labels = []
    
    for lang in langs:
        # 加载隐藏层输出
        pt_model_path = os.path.join(model1_path, "pt_file", task, lang)
        hidden_state = only_first_pt_hidden_states(pt_model_path, model_idx1, device)[num_layers_to_select].cpu().numpy()
        hidden_states.append(hidden_state)
        labels.extend([lang] * hidden_state.shape[0])  # 每个样本对应语言标签


    # 假设最终 X_python: shape (N_python, d_model)
    #           X_java:   shape (N_java, d_model)
    X = np.concatenate(hidden_states, axis=0)
    labels = np.array(labels)

    # 降维到2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # 可视化
    # 分配颜色给每种语言
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))  # 使用颜色映射
    color_map = {lang: colors(idx) for idx, lang in enumerate(unique_labels)}

    plt.figure(figsize=(10, 8))
    for lang in unique_labels:
        idx = labels == lang
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c=[color_map[lang]], label=lang, alpha=0.6)

    plt.title(f'Multi-Language Embedding Space ({task})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    # plt.xlim(-30, 60) 
    # plt.ylim(-40, 20)

    plt.legend(title="Languages")
    plt.savefig(f"simMeasures/pyplot/PCA_PLs/MPLs/{num_layers_to_select}PCA_{model_idx1}_{task}_MPLs.png", format='png', dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":

    # 记录开始时间
    start_time = time.time()    

    device_model = torch.device("cuda:3")  # 第x块GPU


    # 参数设置
    configs = json5.load(open(
        '/newdisk/public/wws/simMeasures/config/config-PCA-PLs.json5'))    # M

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

            # 调用主函数
            print(f"Current work: {task}, Model: {model_idx1}, lang: {langs}")
            main(task, num_layers_to_select, model_path, model_idx1, langs, device_model)
            print(f"Finish work: {task}, Model: {model_idx1}, lang: {langs}")
            print("-"*50)
            print("-"*50)
            print("-"*50)

        # 记录结束时间
        end_time = time.time()
        # 计算并打印程序运行时间
        elapsed_time = (end_time - start_time) / 60
        print(f"Program runtime: {elapsed_time:.2f} mins")    

