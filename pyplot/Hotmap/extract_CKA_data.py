import os
import pandas as pd

def extract_and_save_cka(dir_path, sheet_name, measure, output_file):
    """
    从目录中所有xlsx文件的指定sheet中提取{measure}列，并将其保存到一个新的xlsx文件中。

    参数:
        dir_path (str): 包含xlsx文件的目录路径。
        sheet_name (str): 要提取列的sheet名称。
        output_file (str): 输出xlsx文件的路径。
    """
    # 用于存储指定sheet数据的字典
    sheet_data = {}

    # 遍历目录中的所有文件
    for file in os.listdir(dir_path):
        if file.endswith('.xlsx'):
            file_path = os.path.join(dir_path, file)
            try:
                # 从xlsx文件中读取指定sheet
                df = pd.read_excel(file_path, sheet_name=sheet_name)

                # 检查是否存在“CKA”列
                if measure in df.columns:
                    # 将列名重命名为文件名（去掉扩展名）
                    column_name = os.path.splitext(file)[0]
                    sheet_data[column_name] = df[measure]
                else:
                    print(f"'measure' column not found in sheet '{sheet_name}' of {file}. Skipping.")
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    # 使用收集到的数据创建一个DataFrame
    combined_df = pd.DataFrame(sheet_data)

    # 将合并后的数据保存到输出xlsx文件
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        combined_df.to_excel(writer, sheet_name=f"{measure}_{sheet_name}", index=False)

if __name__ == "__main__":

    # 示例用法
    dir_path = "/newdisk/public/wws/simMeasures/results/final_strategy_tasks_aggre_non_homogeneous_models"  
    sheet_name = "java"  # 
    measure = "CKA"
    output_file = f"/newdisk/public/wws/simMeasures/pyplot/Hotmap/{dir_path.split('/')[-1]}_{measure}_columns_{sheet_name}.xlsx"  # 替换为目标输出文件路径
    extract_and_save_cka(dir_path, sheet_name, measure, output_file)
