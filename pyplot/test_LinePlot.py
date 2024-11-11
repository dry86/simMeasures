import os
import pandas as pd
import matplotlib.pyplot as plt

def find_files(dir_path):
    # 遍历文件夹，寻找以'codeRepair'结尾的xlsx文件
    file_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith("codeRepair.xlsx"):
                file_paths.append(os.path.join(root, file))
    return file_paths

def read_data(file_path):
    # 读取指定列的数据
    df = pd.read_excel(file_path, usecols=["RSA", "CKA", "DisCor", "EOlapScore"])
    return df

def plot_data(dataframes):
    # 设置绘图布局，假设每行显示3个折线图
    n_files = len(dataframes)
    n_cols = 3
    n_rows = (n_files + n_cols - 1) // n_cols  # 确定行数
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    fig.tight_layout(pad=5.0)

    for i, (df, ax) in enumerate(zip(dataframes, axes.flatten())):
        # 为每个文件数据画折线图
        for col in df.columns:
            ax.plot(df.index, df[col], marker='o', label=col)  # 添加标记
        ax.set_title(f"File {i + 1}")
        ax.legend()
        ax.grid(True)
    
    # 隐藏多余的子图
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    plt.savefig("/newdisk/public/wws/simMeasures/pyplot/test_combined_lineplots.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    dir_path = "/newdisk/public/wws/simMeasures/results/final_strategy"  # 替换为你的文件夹路径
    file_paths = find_files(dir_path)
    dataframes = [read_data(file_path) for file_path in file_paths]
    plot_data(dataframes)

main()
