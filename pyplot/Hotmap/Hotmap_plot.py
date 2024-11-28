import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap_from_xlsx(file_path, sheet_name, output_image_path):
    """
    根据给定的xlsx文件中的数据绘制热力图（hotmap）。

    参数:
        file_path (str): xlsx文件路径。
        sheet_name (str): 要读取的sheet名称。
        output_image_path (str): 保存热力图的图像路径。
    """
    try:
        # 从xlsx文件中读取数据
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # 提取每列的前n行和后n行
        n = 10

        data_2n = pd.DataFrame()

        # 遍历每一列，取前n行和后n行
        for col in data.columns:
            front_n = data[col].head(n)  # 前n行
            back_n = data[col].dropna().tail(n)   # 后n行
            data_2n[col] = pd.concat([front_n, back_n]).reset_index(drop=True)  # 合并前n行和后n行

        data_2n = data_2n.transpose()

        # 创建热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(data_2n, vmin=0.5, vmax=1)   # , linewidths=0.5
        
        # 设置标题和标签
        plt.title(f"Heatmap of {sheet_name}", fontsize=14)
        plt.xticks(ticks=range(data_2n.shape[1]), labels=range(1, data_2n.shape[1]+1), fontsize=10)
        plt.xlabel("Layers", fontsize=12)  # Columns
        plt.ylabel("Model pairs", fontsize=12)     # Rows
        
        # 保存图像
        plt.savefig(output_image_path, format='png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Heatmap saved to {output_image_path}")
        
    except Exception as e:
        print(f"Error occurred while generating heatmap: {e}")

if __name__ == "__main__":

    # 示例用法
    file_path = "/newdisk/public/wws/simMeasures/pyplot/Hotmap/final_strategy_tasks_aggre_CKA_columns_java.xlsx"
    sheet_name = "java"
    output_image_path = f"/newdisk/public/wws/simMeasures/pyplot/Hotmap/{file_path.split('/')[-1]}_heatmap_{sheet_name}.png"
    plot_heatmap_from_xlsx(file_path, sheet_name, output_image_path)
