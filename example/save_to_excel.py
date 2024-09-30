import pandas as pd
import os

# 定义保存到Excel的函数
def save_to_excel(data, file_name="/newdisk/public/wws/simMeasures/example/results.xlsx"):
    # 检查文件是否已经存在
    if not os.path.exists(file_name):
        # 如果不存在，创建新的文件，并写入表头
        df = pd.DataFrame(data, columns=["Method", "Score"])
        df.to_excel(file_name, index=False)
    else:
        # 如果文件已经存在，读取现有数据并追加新的数据
        existing_df = pd.read_excel(file_name)
        new_df = pd.DataFrame(data, columns=["Method", "Score"])
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.to_excel(file_name, index=False)

# 打印并保存计算结果的函数
def print_and_save(cal_method, score):
    # 打印结果
    print(f"\t {cal_method}: {score}")
    
    # 准备要保存的内容
    data_to_save = [[cal_method, score]]
    
    # 调用函数保存数据到Excel
    save_to_excel(data_to_save)

# 示例使用
pwcca_mean = 0.85  # 这是一个示例值
print_and_save("PWCCA similarity", pwcca_mean)

# 你可以多次调用 print_and_save 来追加保存多次计算结果
pwcca_mean = 0.90
print_and_save("PWCCA similarity", pwcca_mean)