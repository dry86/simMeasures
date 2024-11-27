import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import replace_keys_with_values

# 配置图像样式
sns.set_style("whitegrid")
# color_palette = ["#038355", "#ffc34e", "#4e79a7", "#f28e2b"]  # 定义四种颜色
font = {'family': 'DejaVu Sans', 'size': 12}
plt.rc('font', **font)

def plot_data_from_excel(dir_path, task_suffix, category, measure_columns, color_palette, mark, save_path):
    # 找到所有以"tasks"结尾的xlsx文件
    excel_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(f"{task_suffix}.xlsx"):
                excel_files.append(os.path.join(root, file))

    # 用于存储符合条件的sheet数据
    data_groups = {"python": [], "cpp": [], "java": []}

    # 遍历每个Excel文件
    for file in excel_files:
        xls = pd.ExcelFile(file)
        
        # 筛选以“language”结尾的sheet
        for sheet_name in xls.sheet_names:
            if sheet_name.endswith("python") or sheet_name.endswith("Python") or sheet_name.endswith("py150"):
                data_groups["python"].append((file, sheet_name))
            elif sheet_name.endswith("cpp") or sheet_name.endswith("CPP"):
                data_groups["cpp"].append((file, sheet_name))
            elif sheet_name.endswith("java") or sheet_name.endswith("javaCorpus"):
                data_groups["java"].append((file, sheet_name))


    # 删除空项并对每组sheet按字母排序
    data_groups = {group: sorted(sheets, key=lambda x: x[1].lower()) for group, sheets in data_groups.items() if sheets}

    # 定义每行3个图像的布局
    for group, sheets in data_groups.items():
        n_files = len(sheets)
        n_cols = 3
        n_rows = (n_files + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        fig.suptitle(f"{task_suffix}: {category} {group.capitalize()}", fontweight='bold', fontsize=16)
        plt.subplots_adjust(hspace=0.3)

        # 绘制每个符合条件的sheet
        for idx, (file, sheet_name) in enumerate(sheets):
            # 读取sheet数据
            df = pd.read_excel(file, sheet_name=sheet_name)
            df = df[measure_columns]

            # 获取子图位置
            ax = axes[idx // n_cols, idx % n_cols] if n_rows > 1 else axes[idx % n_cols]

            # 绘制图像
            x_values = df.index
            for i, col in enumerate(df.columns):
                sns.lineplot(
                    x=x_values, 
                    y=df[col], 
                    color=color_palette[i], 
                    linewidth=2.0, 
                    linestyle='-',
                    marker=mark, 
                    # markerfacecolor="white",
                    # markeredgecolor=color_palette[i], 
                    markeredgewidth=0, 
                    # markersize=4, 
                    label=col, 
                    ax=ax
                )

            # 设置标题、标签和图例
            ax.set_title(f"{replace_keys_with_values(sheet_name)}", fontweight='bold', fontsize=12)   # Todo: 调整这里 设置title
            ax.set_xlabel("Layer", fontsize=12)
            ax.set_ylabel("Values", fontsize=12)
            # ax.legend(loc='lower right', frameon=True, fontsize=10)
            ax.set_xticks(range(0, len(x_values), max(1, len(x_values) // 10)))

            # 设置坐标轴样式
            for spine in ax.spines.values():
                spine.set_edgecolor("#CCCCCC")
                spine.set_linewidth(1.5)

        # 隐藏未使用的子图
        for j in range(idx + 1, n_rows * n_cols):
            fig.delaxes(axes.flatten()[j])

        # # 添加共享图例
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper center', ncol=len(measure_columns), frameon=True, fontsize=10)

        # 保存和显示图像
        # save_dir_task = f"/newdisk/public/wws/simMeasures/pyplot/fig_non_homogeneous_models/{task_suffix}/"
        save_dir_task = os.path.join(save_path, task_suffix)
        if not os.path.exists(save_dir_task):
            os.makedirs(save_dir_task)
        plt.savefig(f"{save_dir_task}/A_{task_suffix}_{category}_lineplots_{group}.png", dpi=300, bbox_inches='tight')

        # save_dir_cate = f"/newdisk/public/wws/simMeasures/pyplot/fig_non_homogeneous_models/{category}/"
        save_dir_cate = os.path.join(save_path, category)
        if not os.path.exists(save_dir_cate):
            os.makedirs(save_dir_cate)
        plt.savefig(f"{save_dir_cate}/B_{task_suffix}_{category}_lineplots_{group}.png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close(fig)


if __name__ == "__main__":

    # 配色方案字典，每个类别对应一个颜色列表
    color_schemes = {
        "Alignment": ["#3b8bba", "#ffa600", "#bc5090", "#58508d", "#ff6361"],  
        "RSM": ["#038355", "#ffc34e", "#4e79a7", "#f28e2b"],  
        "Neighbors": ["#00a5cf", "#f9844a", "#ed6a5a", "#7c878e"],  
        "Topology": ["#1a9988"],  
        "Statistic": ["#ff9f1c", "#2ec4b6", "#e71d36"]  
    }
    marker_schemes = {"Alignment": "p", "RSM": "o", "Neighbors": "^", "Topology": "s", "Statistic": "v"}

    # 分析数据的文件路径
    dir_path = "/newdisk/public/wws/simMeasures/results/final_strategy"
    # save_path = "/newdisk/public/wws/simMeasures/pyplot/figure2"
    save_path = "/newdisk/public/wws/simMeasures/pyplot/figure-textGen_humaneval_instrcut"

    tasks = ["humaneval_finalToken", "mbpp_finalToken", "lineCompletion", "codeSummary-CSearchNet", "codeRepair"]
    # tasks = ["textGen_humaneval", "textGen_MBPP", "line_completion", "codeSummary_CSearchNet", "codeRepair"]
    
    measures_dict = {
    "Alignment": ["OrthProCAN", "OrthAngShape", "AliCosSim", "SoftCorMatch", "HardCorMatch"],  # 
    "RSM": ["RSA", "CKA", "DisCor", "EOlapScore"],
    "Neighbors": ["JacSim", "SecOrdCosSim", "RankSim", "RankJacSim"],
    # "Topology": ["IMD"],
    "Statistic": ["MagDiff", "ConDiff", "UniDiff"]
    }

    tasks = ["humaneval_finalToken", "textGen_humaneval_instrcut", "textGen_humaneval_instrcut2"]
    
    for task in tasks:
        for cate, measure in measures_dict.items():

            plot_data_from_excel(dir_path, task, cate, measure, color_schemes[cate], marker_schemes[cate], save_path)
            # break
