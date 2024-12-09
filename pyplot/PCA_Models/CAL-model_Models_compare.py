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

def reduce_hidden_states(hidden_states, target_dim=150):
    pca = PCA(n_components=target_dim)
    reduced_states = [pca.fit_transform(state) for state in hidden_states]
    return reduced_states

def align_hidden_states(hidden_states):
    max_dim = max(state.shape[1] for state in hidden_states)  # 找到最大的特征维度
    aligned_states = []
    for state in hidden_states:
        if state.shape[1] < max_dim:  # 零填充
            padding = ((0, 0), (0, max_dim - state.shape[1]))
            aligned_state = np.pad(state, padding, mode='constant', constant_values=0)
        else:  # 截断
            aligned_state = state[:, :max_dim]
        aligned_states.append(aligned_state)
    return aligned_states

def main(task, num_layers_to_select, model_paths, language, device):

    hidden_states = []
    labels = []
    
    for model_path in model_paths:
        # 加载隐藏层输出
        model_idx = os.path.basename(model_path)
        pt_model_path = os.path.join(model_path, "pt_file", task, language)
        hidden_state = only_first_pt_hidden_states(pt_model_path, model_idx, device)[num_layers_to_select].cpu().numpy()
        hidden_states.append(hidden_state)
        labels.extend([model_idx] * hidden_state.shape[0])  # 每个样本对应语言标签

    # hidden_states = align_hidden_states(hidden_states)
    hidden_states = reduce_hidden_states(hidden_states, 164)

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

    plt.title(f'Models Embedding Space ({task})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    # plt.xlim(-30, 60) 
    # plt.ylim(-40, 20)

    plt.legend(title="Models")
    plt.savefig(f"simMeasures/pyplot/PCA_Models/reduce-{num_layers_to_select}PCA_{language}_{task}_Models.png", format='png', dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":

    # 记录开始时间
    start_time = time.time()    

    device_model = torch.device("cuda:3")  # 第x块GPU


    # 参数设置
    configs = json5.load(open(
        '/newdisk/public/wws/simMeasures/config/config-PCA-Models.json5'))    # M

    for config in configs:
        task = config.get('task')
        lang = config.get('lang')
        model_paths = config.get('model_paths')
        print(task, lang, model_paths)
    print("-"*50)

    for config in configs:
        tasks = config.get('task')
        lang = config.get('lang')
        model_paths = config.get('model_paths')
        num_layers_to_select = config.get('num_layers_to_select')

        for task in tasks:
            
            # 调用主函数
            print(f"Current work: {task}, Model: {model_paths}, lang: {lang}")
            main(task, num_layers_to_select, model_paths, lang, device_model)
            print(f"Finish work: {task}, Model: {model_paths}, lang: {lang}")
            print("-"*50)
            print("-"*50)
            print("-"*50)

        # 记录结束时间
        end_time = time.time()
        # 计算并打印程序运行时间
        elapsed_time = (end_time - start_time) / 60
        print(f"Program runtime: {elapsed_time:.2f} mins")    

