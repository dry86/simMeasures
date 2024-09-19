import os
import sys
# 将CodeBLEU目录添加到系统路径中
sys.path.append(os.path.abspath("/newdisk/public/wws/simMeasures/CodeBLEU"))

from CodeBLEU import bleu
from CodeBLEU import weighted_ngram_match
from CodeBLEU import syntax_match
from CodeBLEU import dataflow_match

import torch
from getHiddenStates import load_model, get_hidden_states
import numpy as np
import jsonlines
from example.Topology import *
import evaluate


def calculate_PBM_BLEU(pred1, pred2, references):
    # 计算模型1的BLEU分数
    results_model1 = bleu.compute(predictions=pred1, references=references)
    print(f"\t{'模型1的BLEU分数':<20}: {results_model1['bleu']}")

    # 计算模型2的BLEU分数
    results_model2 = bleu.compute(predictions=pred2, references=references)
    print(f"\t{'模型2的BLEU分数':<20}: {results_model2['bleu']}")

    pbmBLUE_score = abs(results_model1['bleu'] - results_model2['bleu'])
    print(f"\t{'pbmBLUE_score':<20}: {pbmBLUE_score:.16f}")

def calculate_codebleu(ref_texts, hyp_texts, lang, params='0.25,0.25,0.25,0.25'):
    """
    计算 CodeBLEU 分数
    :param ref_texts: 参考代码字符串列表，支持多个参考代码（列表内每个元素是一个字符串）
    :param hyp_texts: 生成代码字符串列表
    :param lang: 编程语言 (支持 'java', 'js', 'c_sharp', 'php', 'go', 'python', 'ruby')
    :param params: 权重参数 alpha, beta, gamma, theta（默认 '0.25,0.25,0.25,0.25'）
    :return: CodeBLEU 分数
    """
    
    alpha, beta, gamma, theta = [float(x) for x in params.split(',')]

    # 将单个代码段转为列表形式以进行后续处理
    references = [[ref_texts.strip()]]
    hypothesis = [hyp_texts.strip()]

    assert len(hypothesis) == len(references)

    # 计算 ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    # 计算 weighted ngram match
    keywords = [x.strip() for x in open('/newdisk/public/wws/simMeasures/CodeBLEU/keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 for token in reference_tokens}

    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)]
                                    for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)

    # 计算 syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)

    # 计算 dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)

    print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'.format(
        ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))

    # 计算最终的 CodeBLEU 分数
    code_bleu_score = alpha * ngram_match_score \
                      + beta * weighted_ngram_match_score \
                      + gamma * syntax_match_score \
                      + theta * dataflow_match_score

    # print('CodeBLEU score: ', code_bleu_score)

    return code_bleu_score

def calculate_PBM_codeBLEU(pred1, pred2, ref):
    lang = 'python'
    params = '0.25,0.25,0.25,0.25'

    codebleu_score1 = calculate_codebleu(ref, pred1, lang, params)
    print(f"\t{'CodeBLEU score1':<20}: {codebleu_score1}")
    codebleu_score2 = calculate_codebleu(ref, pred2, lang, params)
    print(f"\t{'CodeBLEU score2':<20}: {codebleu_score2}")

    pbmCodeBLUE_score = abs(codebleu_score1 - codebleu_score2)
    print(f"\t{'pbmCodeBLUE_score':<20}: {pbmCodeBLUE_score:.16f}")



# 指定GPU设备：
device_model1 = torch.device("cuda:0")  # 第x块GPU
device_model2 = torch.device("cuda:1")  # 第y块GPU

# 设置模型和输入
model_7b        = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b"
model_7b_Python = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b-Python"

model1, tokenizer1 = load_model(model_7b, device_model1)
model2, tokenizer2 = load_model(model_7b_Python, device_model2)


# 打开jsonl文件并遍历
file_path = '/newdisk/public/wws/humaneval-x-main/data/python/data/humaneval.jsonl'  # Dataset

# # 加载 BLEU 指标
# bleu = evaluate.load("bleu")
# 如果卡在load函数上, 将下面一行代码在bash上运行,再运行此.py
# export HF_ENDPOINT=https://hf-mirror.com

with jsonlines.open(file_path) as reader:
    for obj in reader:
        task_id = obj.get('task_id')
        prompt = obj.get('prompt')
        refer = obj.get('canonical_solution')
        # prompt = "def fibonacci("
        print(f"Task ID: {task_id}, Prompt: \n{prompt}")
               

        # 输出所有层的CCA分数后，生成Prompt的模型输出
        inputs = tokenizer1(prompt, return_tensors='pt').to(device_model1)
        output_model1 = model1.generate(**inputs, max_length=512)
        generated_text_model1 = tokenizer1.decode(output_model1[0], skip_special_tokens=True)
        
        inputs = tokenizer2(prompt, return_tensors='pt').to(device_model2)
        output_model2 = model2.generate(**inputs, max_length=512)
        generated_text_model2 = tokenizer2.decode(output_model2[0], skip_special_tokens=True)

        # 输出Prompt的模型生成结果
        print("\nGenerated text by CodeLlama-7b:\n")
        # print(generated_text_model1)
        print("\nGenerated text by CodeLlama-7b-Python:\n")
        # print(generated_text_model2)
        calculate_PBM_codeBLEU(generated_text_model1, generated_text_model2, refer)





