# 神经网络表征相似度度量工具

## 项目简介
这是一个用于计算和分析不同神经网络模型表征相似度的工具集。该项目实现了多种相似度度量方法，可以用来比较不同模型的隐藏层表征之间的关系。

## 主要功能
该项目主要用于计算不同模型的隐藏层表征之间的相似度，包含以下几个主要模块：

### 1. 统计特征比较 (cal_Statistic)
- 幅度差异度量 (Magnitude Difference)
- 同心度差异度量 (Concentricity Difference)
- 均匀度差异度量 (Uniformity Difference)

### 2. 拓扑结构分析 (cal_Topology)
- IMD分数计算 (IMD Score)

### 3. 邻域关系分析 (cal_Neighbors)
- Jaccard相似度
- 二阶余弦相似度
- 排序相似度
- 联合排序Jaccard相似度

### 4. 表征相似度矩阵分析 (cal_RSM)
- RSM范数差异
- 表征相似度分析 (RSA)
- 中心核对齐 (CKA)
- 距离相关性
- 特征空间重叠分数
- GULP分析

### 5. 对齐分析 (cal_Alignment)
- 正交Procrustes中心化归一化
- 正交角度形状度量
- 线性回归
- 对齐余弦相似度
- 软相关匹配
- 硬相关匹配
- 置换Procrustes
- Procrustes大小和形状距离

### 6. CCA相关分析
- 典型相关分析 (CCA)
- SVCCA
- PWCCA

## 项目结构


## 使用方法
(简便使用、自动化)
1. 在配置文件中设置相关参数：
   - 任务类型 (task)
   - 模型路径
   - 语言设置 (lang)
   - 需要选择的层数 (num_layers_to_select)

2. 运行主程序：
```bash
python CAL-Integration-non-homogeneous.py
```

## 技术特点
- 支持多种相似度度量方法
- 可以灵活配置比较的模型层数
- 结果自动保存为Parquet、Excel等格式
- 使用装饰器实现运行时间统计

## 支持任务
- 支持代码、NLP、多模态、三大模型任务
   - 代码任务：humaneval、mbpp、CodeCompletion、CodesearchNet、CodeRefinement
   - NLP任务：SST2、MRPC、MNLI
   - 多模态任务：VQAv2、Flickr30k、FHM

## 依赖库
- torch
- numpy
- json5
- tqdm
- rich
- repsim


