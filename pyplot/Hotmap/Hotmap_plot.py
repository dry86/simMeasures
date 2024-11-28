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

def plot_combined_heatmaps(file_path1, sheet_name1, file_path2, sheet_name2, output_image_path):
    """
    生成两个热力图子图，放置在一个图中，上下排列。

    参数:
        file_path1 (str): 第一个xlsx文件路径。
        sheet_name1 (str): 第一个xlsx文件的sheet名称。
        file_path2 (str): 第二个xlsx文件路径。
        sheet_name2 (str): 第二个xlsx文件的sheet名称。
        output_image_path (str): 保存合并热力图的图像路径。
    """
    try:
        # 读取两个文件的数据
        data_2n_1 = plot_heatmap_from_xlsx(file_path1, sheet_name1, output_image_path)
        data_2n_2 = plot_heatmap_from_xlsx(file_path2, sheet_name2, output_image_path)
        
        if data_2n_1 is None or data_2n_2 is None:
            print("Error in generating heatmaps. Exiting...")
            return

        # 创建一个 2x1 的子图布局
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # 绘制第一个热力图: 同源
        sns.heatmap(data_2n_1, vmin=0.5, vmax=1, ax=axs[0])  # 第一张子图
        axs[0].set_title(f"Homogeneous_models ({sheet_name1})", fontsize=14)
        axs[0].set_xlabel("Layers", fontsize=12)
        axs[0].set_ylabel("Model pairs", fontsize=12)

        # 绘制第二个热力图: 非同源
        sns.heatmap(data_2n_2, vmin=0.5, vmax=1, ax=axs[1])  # 第二张子图
        axs[1].set_title(f"Non_homogeneous_models ({sheet_name2})", fontsize=14)
        axs[1].set_xlabel("Layers", fontsize=12)
        axs[1].set_ylabel("Model pairs", fontsize=12)

        # 保存图像
        plt.tight_layout()
        plt.savefig(output_image_path, format='png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Combined heatmap saved to {output_image_path}")

    except Exception as e:
        print(f"Error occurred while generating combined heatmaps: {e}")

if __name__ == "__main__":

    # 示例用法
    file_path1 = "/newdisk/public/wws/simMeasures/pyplot/Hotmap/final_strategy_tasks_aggre_CKA_columns_java.xlsx"
    sheet_name1 = "java"
    file_path2 = "/newdisk/public/wws/simMeasures/pyplot/Hotmap/final_strategy_tasks_aggre_non_homogeneous_models_CKA_columns_java.xlsx"
    sheet_name2 = "java"
    output_image_path = f"/newdisk/public/wws/simMeasures/pyplot/Hotmap/combined_heatmap.png"

    plot_combined_heatmaps(file_path1, sheet_name1, file_path2, sheet_name2, output_image_path)