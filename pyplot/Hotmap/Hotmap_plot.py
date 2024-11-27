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
        data = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
        
        # 提取每列的前n行和后n行
        n = 10  # 替换为所需的行数
        data_back = data.apply(lambda col: col[col.last_valid_index() - n + 1: col.last_valid_index() + 1], axis=0) # data.apply(lambda col: col.dropna().tail(n), axis=0)  # 每列的最后n个有效值
        data = pd.concat([data.head(n), data_back]).drop_duplicates()

        data = data.transpose()

        # 创建热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, cmap="YlGnBu", cbar=True, annot=False, linewidths=0.5)
        
        # 设置标题和标签
        plt.title(f"Heatmap of {sheet_name}", fontsize=14)
        plt.xticks(ticks=range(data.shape[1]), labels=range(0, data.shape[1]), fontsize=10)
        plt.xlabel("Columns", fontsize=12)
        plt.ylabel("Rows", fontsize=12)
        
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
