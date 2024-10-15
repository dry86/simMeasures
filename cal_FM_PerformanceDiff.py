import os
import sys
import torch
import jsonlines
import numpy as np
# 将CodeBLEU目录添加到系统路径中
sys.path.append(os.path.abspath("/newdisk/public/wws/simMeasures/CodeBLEU"))
sys.path.append(os.path.abspath("/newdisk/public/wws/simMeasures/bleu-evaluator"))
from CodeBLEU import bleu
from CodeBLEU import weighted_ngram_match
from CodeBLEU import syntax_match
from CodeBLEU import dataflow_match

from utils import load_model_and_tokenizer
from utils import compute_bleu

# BLEU
def calculate_PBM_BLEU(pred1, pred2, references):
    max_order = 4
    smooth = False  # 在此参数的设定下, 计算结果 = codeBLEU.ngram_match_score

    # references = [references.strip().split()]
    # pred1 = pred1.strip().split()
    # pred2 = pred2.strip().split()

    references = [[references.strip()]]
    hypo_pred1 = [pred1.strip()]
    hypo_pred2 = [pred2.strip()]

    assert len(hypo_pred1) == len(references)
    assert len(hypo_pred2) == len(references)

    # 计算 ngram match (BLEU)
    tokenized_hyps_pred1 = [x.split() for x in hypo_pred1]
    tokenized_hyps_pred2 = [x.split() for x in hypo_pred2]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    # 计算模型1的BLEU分数
    bleu_score1, _, _, _, _, _  = compute_bleu(tokenized_refs, tokenized_hyps_pred1, max_order, smooth)
    print(f"\t{'模型1的BLEU分数':<20}: {bleu_score1}")

    # 计算模型2的BLEU分数
    bleu_score2, _, _, _, _, _  = compute_bleu(tokenized_refs, tokenized_hyps_pred2, max_order, smooth)
    print(f"\t{'模型2的BLEU分数':<20}: {bleu_score2}")

    pbmBLEU_score = abs(bleu_score1 - bleu_score2)
    print(f"\t{'pbmBLEU_score':<20}: {pbmBLEU_score:.16f}")

    return bleu_score1, bleu_score2, pbmBLEU_score

# codeBLEU
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

    return codebleu_score1, codebleu_score2, pbmCodeBLUE_score

def main(model_1, model_2, file_path, device1, device2):

    model1, tokenizer1 = load_model_and_tokenizer(model_1, device1)
    model2, tokenizer2 = load_model_and_tokenizer(model_2, device2)

    bleu_model1 = []
    bleu_model2 = []
    bleu_diff = []

    codebleu_model1 = []
    codebleu_model2 = []
    codebleu_diff = []  

    with jsonlines.open(file_path) as reader:
        for obj in reader:
            task_id = obj.get('task_id')
            prompt = obj.get('prompt')
            refer = prompt + obj.get('canonical_solution')
            print(f"Task ID: {task_id}")
                
            # 输出所有层的CCA分数后，生成Prompt的模型输出
            inputs1 = tokenizer1(prompt, return_tensors='pt').to(device1)
            inputs2 = tokenizer2(prompt, return_tensors='pt').to(device2)
            with torch.no_grad():
                output_model1 = model1.generate(**inputs1, max_length=512)
                output_model2 = model2.generate(**inputs2, max_length=512)
                generated_text_model1 = tokenizer1.decode(output_model1[0], skip_special_tokens=True)
                generated_text_model2 = tokenizer2.decode(output_model2[0], skip_special_tokens=True)

            # 输出Prompt的模型生成结果
            # print("\nGenerated text by model_1:\n")
            # print(generated_text_model1)
            # print("\nGenerated text by model_2:\n")
            # print(generated_text_model2)
            bleu_score1, bleu_score2, bleu_scoreDiff = calculate_PBM_BLEU(generated_text_model1, generated_text_model2, refer)
            codebleu_score1, codebleu_score2, codebleu_scoreDiff = calculate_PBM_codeBLEU(generated_text_model1, generated_text_model2, refer)

            bleu_model1.append(bleu_score1)
            bleu_model2.append(bleu_score2)
            bleu_diff.append(bleu_scoreDiff)

            codebleu_model1.append(codebleu_score1)
            codebleu_model2.append(codebleu_score2)
            codebleu_diff.append(codebleu_scoreDiff)
            print("------------------------------------------")

    print(f"bleu_model1: {np.mean(bleu_model1)}")
    print(f"bleu_model2: {np.mean(bleu_model2)}")
    print(f"bleu_diff: {np.mean(bleu_diff)}")

    print(f"codebleu_model1: {np.mean(codebleu_model1)}")
    print(f"codebleu_model2: {np.mean(codebleu_model2)}")
    print(f"codebleu_diff: {np.mean(codebleu_diff)}")



if __name__ == "__main__":

    # 指定GPU设备：
    device_model1 = torch.device("cuda:2")  # 第x块GPU
    device_model2 = torch.device("cuda:3")  # 第y块GPU

    # device_model1 = 'cpu'
    # device_model2 = 'cpu'

    # 设置模型和输入
    model_1 = "/newdisk/public/wws/model_dir/codellama/codeLlama-7b"
    model_2 = "/newdisk/public/wws/model_dir/codellama/CodeLlama-7b-Instruct" 

    # 打开jsonl文件并遍历
    file_path = '/newdisk/public/wws/humaneval-x-main/data/python/data/humaneval.jsonl'  # Dataset

    main(model_1, model_2, file_path, device_model1, device_model2)