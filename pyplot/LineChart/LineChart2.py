import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 配置图像样式
sns.set_style("whitegrid")
color_palette = ["#038355", "#ffc34e", "#4e79a7", "#f28e2b"]  # 定义四种颜色
font = {'family': 'DejaVu Sans', 'size': 12}
plt.rc('font', **font)

def plot_data_from_excel(dir_path):
    # 找到所有以"codeRepair"结尾的xlsx文件
    excel_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith("codeRepair.xlsx"):    # 设置要分析的数据集
                excel_files.append(os.path.join(root, file))

    # 用于存储符合条件的sheet数据
    data_groups = {"python": [], "java": []}

    # 遍历每个Excel文件
    for file in excel_files:
        xls = pd.ExcelFile(file)
        
        # 筛选以“python”或“java”结尾的sheet
        for sheet_name in xls.sheet_names:
            if sheet_name.endswith("python"):
                data_groups["python"].append((file, sheet_name))
            elif sheet_name.endswith("java"):
                data_groups["java"].append((file, sheet_name))

    # 定义每行3个图像的布局
    for group, sheets in data_groups.items():
        n_files = len(excel_files)
        n_cols = 3
        n_rows = (n_files + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        fig.suptitle("Line Plots of Excel Files", fontweight='bold', fontsize=16)
        plt.subplots_adjust(hspace=0.3)

    # 遍历每个Excel文件并绘制折线图
    for idx, file in enumerate(excel_files):
        # 读取数据
        df = pd.read_excel(file)
        df = df[["RSA", "CKA", "DisCor", "EOlapScore"]]  # 只选择需要的四列

        # 获取子图位置
        ax = axes[idx // n_cols, idx % n_cols] if n_rows > 1 else axes[idx % n_cols]

        # 绘制四列数据的折线图
        x_values = df.index
        for i, col in enumerate(df.columns):
            sns.lineplot(x=x_values, y=df[col], color=color_palette[i], linewidth=2.0, linestyle='-',
                         marker="o", markeredgewidth=1, markerfacecolor="white", markeredgecolor=color_palette[i], markersize=4, # markeredgecolor="white", markeredgewidth=1.5, markersize=4,
                         label=col, ax=ax)

        # 设置标题、标签和图例
        ax.set_title(f"{os.path.basename(file)}", fontweight='bold', fontsize=14)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Values", fontsize=12)
        ax.legend(loc='best', frameon=True, fontsize=10)
        ax.set_xticks(range(0, len(x_values), max(1, len(x_values) // 10)))

        # 设置坐标轴样式
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")
            spine.set_linewidth(1.5)

    # 隐藏未使用的子图
    for j in range(idx + 1, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    # 保存和显示图像
    plt.savefig("/newdisk/public/wws/simMeasures/pyplot/combined_lineplots.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    # 使用示例
    dir_path = "/newdisk/public/wws/simMeasures/results/final_strategy"  # 替换为实际的文件夹路径
    plot_data_from_excel(dir_path)
