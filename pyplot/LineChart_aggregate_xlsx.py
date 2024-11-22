import os
import pandas as pd
import numpy as np


def process_subdir(subdir_path, output_dir, tasks):
    # 获取当前子文件夹名称作为输出文件的文件名
    subdir_name = os.path.basename(subdir_path)
    output_file = os.path.join(output_dir, f"{subdir_name}.xlsx")
    
    # 用于存储各个分类的DataFrame
    data_groups = {"python": [], "cpp": [], "java": []}
    
    # 遍历subdir_path文件夹中的所有xlsx文件
    for file in os.listdir(subdir_path):
        if file.endswith(".xlsx"):
            file_path = os.path.join(subdir_path, file)
            
            # 检查文件名是否以tasks中的某个任务名结尾
            if any(file.endswith(f"{task}.xlsx") for task in tasks):
                # 读取Excel文件中的所有sheet_name
                all_sheets = pd.ExcelFile(file_path).sheet_names
                
                for sheet in all_sheets:
                    # 将符合条件的sheet分组
                    if "python" in sheet.lower() or "py150" in sheet.lower():
                        data_groups["python"].append(pd.read_excel(file_path, sheet_name=sheet))
                    elif "cpp" in sheet.lower():
                        data_groups["cpp"].append(pd.read_excel(file_path, sheet_name=sheet))
                    elif "java" in sheet.lower() or "javaCorpus" in sheet.lower():
                        data_groups["java"].append(pd.read_excel(file_path, sheet_name=sheet))
    
    if not any(data_groups.values()):
        print(f"No valid sheets to process in {subdir_path}. Skipping.")
        return

    # 使用ExcelWriter来保存不同语言的DataFrame到同一个文件的不同sheet中
    with pd.ExcelWriter(output_file) as writer:
        # 为每个组中的DataFrame计算平均值并保存
        for language, dfs in data_groups.items():
            if dfs:
                # 逐元素相加并求平均
                sum_df = dfs[0].copy()  # 创建一个与第一个DataFrame结构相同的副本，用于存储求和
                for df in dfs[1:]:
                    sum_df += df  # 逐元素相加
                
                averaged_df = sum_df / len(dfs)  # 逐元素求平均

                # 保存处理后的DataFrame到Excel文件
                averaged_df.to_excel(writer, sheet_name=language, index=False)
                print(f"Processed and saved {language} sheet in {output_file}")

def process_all_subdirs(base_dir, output_dir, tasks):
    # 遍历base_dir文件夹下的所有子文件夹
    for subdir_name in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir_name)
        
        # 如果是文件夹，调用处理函数
        if os.path.isdir(subdir_path):
            process_subdir(subdir_path, output_dir, tasks)

if __name__ == "__main__":

    # 任务名列表
    tasks = ["humaneval_finalToken", "mbpp_finalToken", "lineCompletion", "codeSummary-CSearchNet", "codeRepair"]

    # 设置输入文件夹路径和输出文件夹路径
    base_dir = '/newdisk/public/wws/simMeasures/results/final_strategy'  # base目录，包含多个subdir_path
    output_dir = '/newdisk/public/wws/simMeasures/results/final_strategy_tasks_aggre'  # 输出目录

    # 调用函数处理所有子文件夹
    process_all_subdirs(base_dir, output_dir, tasks)
