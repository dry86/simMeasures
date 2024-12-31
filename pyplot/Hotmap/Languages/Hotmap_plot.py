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

        return data_2n

    except Exception as e:
        print(f"Error occurred while generating heatmap: {e}")
        return None

def plot_combined_heatmaps(file_path1, sheet_name1, output_image_path):
    """
    生成一个热力图。

    参数:
        file_path1 (str): xlsx文件路径。
        sheet_name1 (str): xlsx文件的sheet名称。
        output_image_path (str): 保存热力图的图像路径。
    """
    try:
        # 读取文件的数据
        data_2n_1 = plot_heatmap_from_xlsx(file_path1, sheet_name1, output_image_path)
        
        if data_2n_1 is None:
            print("Error in generating heatmap. Exiting...")
            return

        # 创建一个图
        plt.figure(figsize=(8, 16))

        # 绘制热力图
        sns.heatmap(data_2n_1, cmap="YlGnBu")  # vmin=0.5, vmax=1, 可根据需要调整
        plt.title(f"Heatmap ({sheet_name1})", fontsize=14)
        plt.xlabel("Layers", fontsize=12)
        plt.ylabel("Langs pairs", fontsize=12)

        # 保存图像
        plt.tight_layout()
        plt.savefig(output_image_path, format='png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Heatmap saved to {output_image_path}")

    except Exception as e:
        print(f"Error occurred while generating heatmap: {e}")


if __name__ == "__main__":

    # 示例用法
    model_idx = "starcoder2-7b" # "codeLlama-7b"  # "Qwen2.5-Coder-7B"
    # model_idx = "Qwen2.5-Coder-7B"
    sheet_name1 = "textGen_humaneval"

    file_path1 = f"/newdisk/public/wws/simMeasures/pyplot/Hotmap/Languages/xlsx/{model_idx}-MPLs.xlsx"
    output_image_path = f"/newdisk/public/wws/simMeasures/pyplot/Hotmap/Languages/png/MPLs-{model_idx}_{sheet_name1}_Hotmap.png"

    plot_combined_heatmaps(file_path1, sheet_name1, output_image_path)