import os
import torch
import pandas as pd
import jsonlines
import collections
import math
from openpyxl import load_workbook
from transformers import AutoTokenizer, AutoModelForCausalLM

# TodoList: 数据集的输入 构建一个Class  从类的成员函数中获取模式
import jsonlines

def extract_prompts(data_file_path, mode):
    """
    通用函数，用于从文件中提取Prompts。

    参数:
        data_file_path (str): 数据文件的路径。
        mode (str): 解析模式，可选值：
            - 'textGen_MBPP': 提取'text'字段。
            - 'textGen_humaneval': 提取'prompt'字段。
            - 'codeSummary_CSearchNet': 提取'code'字段，并添加固定前缀。
            - 'codeRepair': 从纯文本文件逐行读取，并添加固定前缀。
            - 'line_completion': 提取'input'字段。
    
    返回:
        list: 提取的Prompt列表。
    """
    # 定义模式与字段及固定前缀的映射
    mode_config = {
        'textGen_MBPP': {'field': 'text', 'prefix': None, 'key': 'task_id'},
        'textGen_humaneval': {'field': 'prompt', 'prefix': None, 'key': 'task_id'},
        'codeSummary_CSearchNet': {'field': 'code', 'prefix': "Please describe the functionality of the method: ", 'key': 'repo'},
        'codeRepair': {'field': None, 'prefix': "Please fix the bug in the following code: ", 'key': None},
        'line_completion': {'field': 'input', 'prefix': None, 'key': 'id'}
    }
    
    if mode not in mode_config:
        raise ValueError(f"Invalid mode: {mode}. Available modes: {list(mode_config.keys())}")
    
    config = mode_config[mode]
    prompts = []

    if mode == 'codeRepair':
        # 特殊处理纯文本文件
        with open(data_file_path, "r") as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        
        for i, line in enumerate(lines, start=1):
            prompt = config['prefix'] + line
            prompts.append((i, prompt))
    else:
        # 通用处理jsonlines文件
        with jsonlines.open(data_file_path) as reader:
            for obj in reader:
                task_id = obj.get(config['key']) if config['key'] else None
                content = obj.get(config['field'], "")
                prompt = (config['prefix'] or "") + content
                prompts.append((task_id, prompt))
    
    return prompts


# TodoList: analysis_max_token

def combine_names(name1, name2, lang, max_length=31):
    def shorten_name(name):
      # 定义名称缩写的条件
      if "codeLlama-7b" in name:
          return name.replace("codeLlama-7b", "cL7b")
      elif "dsc-7b-base-v1.5" in name:
          return name.replace("dsc-7b-base-v1.5", "dsc7b")
      elif "Qwen2.5-Coder-7B" in name:
          return name.replace("Qwen2.5-Coder-7B", "QwC7b")
      # elif "codeShell-7b" in name:
      #     return name.replace("codeShell-7b", "cSh7b")
      # 根据需要添加更多缩写规则
      return name  # 如果不符合任何条件，返回原始名称

    # 依次缩写每个名称
    name1_short = shorten_name(name1)
    name2_short = shorten_name(name2)
    # name3_short = shorten_name(name3)
    
    combined_name = f"{name1_short} vs {name2_short} {lang}"
    
    # 如果缩写后的名称仍然超长，进行截取
    if len(combined_name) > max_length:
        combined_name = lang
    
    return combined_name

# 定义模型加载函数
def load_model_and_tokenizer(model_path: str, device: torch.device):
    """
    加载预训练的模型和分词器，并将模型加载到指定设备上。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        pad_token_id=tokenizer.pad_token_id,
        torch_dtype=torch.float32
    ).to(device)
    return model, tokenizer

class ResultSaver:
    def __init__(self, sheet, file_name):
        self.file_name = file_name
        self.sheet = sheet

    # 定义保存到Excel的函数，支持指定工作表
    def save_to_excel(self, cal_method, score, row):
        file_name = self.file_name
        sheet = self.sheet

        # 如果文件不存在，创建新文件并写入数据
        if not os.path.exists(file_name):
            directory = os.path.dirname(file_name)
            # 检查目录是否存在，如果不存在则创建
            if not os.path.exists(directory):
                os.makedirs(directory)

            # 创建一个新的 DataFrame，并保存到指定工作表中
            df = pd.DataFrame({cal_method: [None] * row})
            df.loc[row - 1, cal_method] = score
            with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet, index=False)
        else:
            # 文件已存在，加载现有文件
            book = load_workbook(file_name)
            
            # 检查工作表是否存在
            if sheet in book.sheetnames:
                df_existing = pd.read_excel(file_name, sheet_name=sheet)
            else:
                # 如果工作表不存在，创建一个新的空DataFrame
                df_existing = pd.DataFrame()

            # 如果列已经存在，则追加数据
            if cal_method in df_existing.columns:
                if len(df_existing) < row:  # 扩展行数
                    df_existing = df_existing.reindex(list(range(row)))
                df_existing.loc[row - 1, cal_method] = score
            else:
                # 如果列不存在，创建新列并填充数据
                df_existing[cal_method] = [None] * len(df_existing)
                if len(df_existing) < row:  # 同样检查是否需要扩展行数
                    df_existing = df_existing.reindex(list(range(row)))
                df_existing.loc[row - 1, cal_method] = score
            
            # 使用 openpyxl 引擎以附加模式写入工作簿，而不会覆盖其他工作表
            with pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                # writer.book = book  # 使用已加载的工作簿
                df_existing.to_excel(writer, sheet_name=sheet, index=False)

            # 重新打开文件，设置单元格格式为数字类型，避免科学计数法
            book = load_workbook(file_name)
            sheet_to_format = book[sheet]
            
            # 设置单元格格式为浮点型数字显示
            for row_cells in sheet_to_format.iter_rows(min_row=2, max_row=row+1, min_col=1, max_col=len(df_existing.columns)):
                for cell in row_cells:
                    if isinstance(cell.value, float):  # 只对小数类型的单元格进行格式设置
                        cell.number_format = '0.0000000000000000'  # 设置足够的位数显示小数点后的所有位数

            # 保存工作簿
            book.save(file_name)
        
    # 打印并保存计算结果的函数
    def print_and_save(self, cal_method, score, row):
        # 打印结果
        print(f"\t {cal_method}: {score}")
        
        # 调用函数保存数据到指定工作表中
        self.save_to_excel(cal_method, score, row + 1)

def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts

def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)


if __name__ == "__main__":
    # 示例使用
  # saver = ResultSaver(file_name="/newdisk/public/wws/simMeasures/results/codellama_7b_and_7b_python/test.xlsx", sheet="Metrics")
  # pwcca_mean = 0.1575322875319305  # 这是一个示例值
  # saver.print_and_save("PWCCA similarity", pwcca_mean, row=1)

  # pwcca_mean = 0.897656342134534
  # saver.print_and_save("PWCCA similarity", pwcca_mean, row=2)
  # saver = ResultSaver(file_name="/newdisk/public/wws/simMeasures/results/codellama_7b_and_7b_python/test.xlsx", sheet="Metrics1")
  # # 示例添加另一个cal_method
  # cosine_sim = 0.789456123456789
  # saver.print_and_save("Cosine similarity", cosine_sim, row=1)

  data_file_path = "/newdisk/public/wws/Dataset/CodeSearchNet/dataset/java/test.jsonl"
  prompts = extract_prompts(data_file_path, mode='codeSummary_CSearchNet', prompt_prefix="Please describe the functionality of the method: ")
  print(prompts[1])
