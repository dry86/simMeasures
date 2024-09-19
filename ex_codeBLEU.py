import os
import sys
# 将CodeBLEU目录添加到系统路径中
sys.path.append(os.path.abspath("/newdisk/public/wws/simMeasures/CodeBLEU"))

from CodeBLEU import bleu
from CodeBLEU import weighted_ngram_match
from CodeBLEU import syntax_match
from CodeBLEU import dataflow_match

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

    # 直接使用传入的参考代码和生成代码
    pre_references = [refs.strip().splitlines() for refs in ref_texts]
    hypothesis = [hyp.strip() for hyp in hyp_texts]

    # for i in range(len(pre_references)):
    #     assert len(hypothesis) == len(pre_references[i])

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)

    # assert len(references) == len(pre_references) * len(hypothesis)

    # 计算 ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    # 计算 weighted ngram match
    keywords = [x.strip() for x in open('keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]

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

    print('CodeBLEU score: ', code_bleu_score)

    return code_bleu_score

# 示例使用
if __name__ == "__main__":

    # 示例用法
    ref_texts = ["def add(a, b):\n    return a + b"]
    hyp_texts = ["def add(a, b):\n    return a + b"]
    lang = 'python'
    params = '0.25,0.25,0.25,0.25'

    codebleu_score = calculate_codebleu(ref_texts, hyp_texts, lang, params)
    print(f'Final CodeBLEU score: {codebleu_score}')