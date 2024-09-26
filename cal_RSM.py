import torch
from getHiddenStates import load_model, get_hidden_states
from sklearn.cross_decomposition import CCA
import numpy as np
import jsonlines
from example.RSM import *

class CKA_np:
    def __init__(self):
        pass
    
    def cka(self, x1, x2, debiased=False):
        """Compute the CKA similarity between two sets of activations."""
        x1 = self.gram_linear(self.rearrange_activations(x1))
        x2 = self.gram_linear(self.rearrange_activations(x2))
        similarity = self._cka(x1, x2, debiased=debiased)
        return similarity

    def rearrange_activations(self, activations):
        """Flatten the activations into a batch_size x num_features format."""
        batch_size = activations.shape[0]
        flat_activations = activations.reshape(batch_size, -1)
        return flat_activations

    def gram_linear(self, x):
        """Compute Gram (kernel) matrix for a linear kernel.

        Args:
            x: A num_examples x num_features matrix of features.

        Returns:
            A num_examples x num_examples Gram matrix of examples.
        """
        return x.dot(x.T)

    def center_gram(self, gram, unbiased=False):
        """Center a symmetric Gram matrix.

        This is equvialent to centering the (possibly infinite-dimensional) features
        induced by the kernel before computing the Gram matrix.

        Args:
            gram: A num_examples x num_examples symmetric matrix.
            unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
            estimate of HSIC. Note that this estimator may be negative.

        Returns:
            A symmetric matrix with centered columns and rows.
        """
        if not np.allclose(gram, gram.T):
            raise ValueError('Input must be a symmetric matrix.')
        gram = gram.copy()

        if unbiased:
            # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
            # L. (2014). Partial distance correlation with methods for dissimilarities.
            # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
            # stable than the alternative from Song et al. (2007).
            n = gram.shape[0]
            np.fill_diagonal(gram, 0)
            means = np.sum(gram, axis=0, dtype=np.float64) / (n - 2)
            means -= np.sum(means) / (2 * (n - 1))
            gram -= means[:, None]
            gram -= means[None, :]
            np.fill_diagonal(gram, 0)
        else:
            means = np.mean(gram, axis=0, dtype=np.float64)
            means -= np.mean(means) / 2
            gram -= means[:, None]
            gram -= means[None, :]

        return gram

    def _cka(self, gram_x, gram_y, debiased=False):
        """Compute CKA.

        Args:
            gram_x: A num_examples x num_examples Gram matrix.
            gram_y: A num_examples x num_examples Gram matrix.
            debiased: Use unbiased estimator of HSIC. CKA may still be biased.

        Returns:
            The value of CKA between X and Y.
        """
        gram_x = self.center_gram(gram_x, unbiased=debiased)
        gram_y = self.center_gram(gram_y, unbiased=debiased)

        # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
        # n*(n-3) (unbiased variant), but this cancels for CKA.
        # Calculate the scaled HSIC
        scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

        # Normalization
        normalization_x = np.linalg.norm(gram_x)
        normalization_y = np.linalg.norm(gram_y)
        
        # Return CKA similarity
        return scaled_hsic / (normalization_x * normalization_y)


def calculate_RSM(acts1, acts2, idx):
    print(f"Layer {idx}, shape: {acts1.shape}:")
    # 计算rsm相似度
    similarity_rsm = cal_RSM_Norm_Difference(acts1, acts2)
    print(f"\t{'Representational Similarity (Frobenius norm difference)':<60}: {similarity_rsm:.16f}")

    # 计算rsa相似度
    similarity_rsa = cal_RSA(acts1, acts2)
    print(f"\t{'Representational Similarity (Pearson correlation)':<60}: {similarity_rsa:.16f}")

    # 计算cka相似度
    similarity_cka = cal_cka(acts1, acts2)
    print(f"\t{'CKA Similarity':<60}: {similarity_cka:.16f}")
    cka_similarity = cka_calculator.cka(acts1, acts2)
    print(f"\t{'CKA Similarity_new':<60}: {cka_similarity:.16f}")

    # 计算dCor距离相关性
    dcor_score = cal_distance_correlation(acts1, acts2)
    print(f"\t{'Distance Correlation':<60}: {dcor_score:.16f}")

    # 计算 Normalized Bures Similarity
    nbs_score = cal_bures_similarity(acts1, acts2)
    print(f"\t{'Normalized Bures Similarity':<60}: {np.real(nbs_score)}")

    # 计算 Eigenspace Overlap Score
    eos_score = cal_eigenspace_overlap_score(acts1, acts2)
    print(f"\t{'Eigenspace Overlap Score (Normalized)':<60}: {eos_score}")

    # 计算 Unified Linear Probing (GULP).
    glup_score = cal_gulp_measure(acts1, acts2)
    print(f"\t{'Unified Linear Probing (GULP)':<60}: {glup_score}")

    # 计算 Riemannian Distance
    riemann_dist = cal_riemannian_distance(acts1, acts2)
    print(f"\t{'Riemannian Distance':<60}: {riemann_dist}")


# 指定GPU设备：
device_model1 = torch.device("cuda:2")  # 第x块GPU
device_model2 = torch.device("cuda:3")  # 第y块GPU

# 设置模型和输入
model_7b        = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b"
model_7b_Python = "/newdisk/public/wws/text-generation-webui/models/codeLlama-7b-Python"

model1, tokenizer1 = load_model(model_7b, device_model1)
model2, tokenizer2 = load_model(model_7b_Python, device_model2)

cka_calculator = CKA_np()

# 打开jsonl文件并遍历
file_path = '/newdisk/public/wws/humaneval-x-main/data/python/data/humaneval.jsonl'  # Dataset

with jsonlines.open(file_path) as reader:
    for obj in reader:
        task_id = obj.get('task_id')
        prompt = obj.get('prompt')
        # prompt = "def fibonacci("
        print(f"Task ID: {task_id}, Prompt: \n{prompt}")
        
        # layer_indices = [1, -2]  # 倒数第二层和第二层

        # 获取隐藏层矩阵
        hidden_states_model1 = get_hidden_states(model1, tokenizer1, prompt, device_model1)
        hidden_states_model2 = get_hidden_states(model2, tokenizer2, prompt, device_model2)

        
        # 获取模型的总层数
        num_layers = len(hidden_states_model1)

        # 获取每一层的CCA相关性得分
        
        for i in range(num_layers):
            acts1 = hidden_states_model1[i].reshape(-1, hidden_states_model1[i].shape[-1])
            acts2 = hidden_states_model2[i].reshape(-1, hidden_states_model2[i].shape[-1])
            # print(f"hidden layer shape: {acts1.shape}")
            calculate_RSM(acts1, acts2, i)
            

        # 输出所有层的CCA分数后，生成Prompt的模型输出
        inputs = tokenizer1(prompt, return_tensors='pt').to(device_model1)
        output_model1 = model1.generate(**inputs, max_length=512)
        generated_text_model1 = tokenizer1.decode(output_model1[0], skip_special_tokens=True)
        
        inputs = tokenizer2(prompt, return_tensors='pt').to(device_model2)
        output_model2 = model2.generate(**inputs, max_length=512)
        generated_text_model2 = tokenizer2.decode(output_model2[0], skip_special_tokens=True)

        # 输出Prompt的模型生成结果
        print("\nGenerated text by CodeLlama-7b:\n")
        print(generated_text_model1)
        print("\nGenerated text by CodeLlama-7b-Python:\n")
        print(generated_text_model2)