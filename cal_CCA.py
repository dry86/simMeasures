import torch
from getHiddenStates import load_model, get_hidden_states
import numpy as np
import jsonlines
from example import cca_core


def rearrange_activations(activations):
    batch_size = activations.shape[0]
    flat_activations = activations.view(batch_size, -1)
    return flat_activations

def cal_cca_torch(x1, x2):
    x1_flat, x2_flat = rearrange_activations(x1), rearrange_activations(x2)

    q1, _ = torch.qr(x1_flat)
    q2, _ = torch.qr(x2_flat)

    cca = (torch.norm(q2.T @ q1)) ** 2 / q1.shape[1]

    print(f"\tCCA_torch similarity: {cca}")

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


def calculate_cca(acts1, acts2, idx):
    
    print(f"Layer {idx}, shape: {acts1.shape}:")
    results = cca_core.get_cca_similarity(acts1, acts2, epsilon=1e-6, verbose=False)

    b1 = np.random.randn(*acts1.shape)
    b2 = np.random.randn(*acts2.shape)
    baseline = cca_core.get_cca_similarity(b1, b2, epsilon=1e-6, verbose=False)
    print("\tBaseline Mean CCA similarity", np.mean(baseline["cca_coef1"]))
    print(f"\tMean CCA similarity: {np.mean(results["cca_coef1"])}")


    # Results using SVCCA keeping 20 dims

    # Mean subtract activations
    cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    svacts1 = np.dot(s1[:20]*np.eye(20), V1[:20])
    svacts2 = np.dot(s2[:20]*np.eye(20), V2[:20])

    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-6, verbose=False)

    # Mean subtract baseline activations
    cb1 = b1 - np.mean(b1, axis=0, keepdims=True)
    cb2 = b2 - np.mean(b2, axis=0, keepdims=True)

    # Perform SVD
    Ub1, sb1, Vb1 = np.linalg.svd(cb1, full_matrices=False)
    Ub2, sb2, Vb2 = np.linalg.svd(cb2, full_matrices=False)

    svb1 = np.dot(sb1[:20]*np.eye(20), Vb1[:20])
    svb2 = np.dot(sb2[:20]*np.eye(20), Vb2[:20])

    svcca_baseline = cca_core.get_cca_similarity(svb1, svb2, epsilon=1e-6, verbose=False)
    print("\tBaseline SVCCA similarity: ", np.mean(svcca_baseline["cca_coef1"]), 
          "\n\tSVCCA similarity: ", np.mean(svcca_results["cca_coef1"]))

    pwcca_mean, w, _ = cca_core.compute_pwcca(acts1, acts2, epsilon=1e-6)
    pwcca_baseline, wb, _ = cca_core.compute_pwcca(b1, b2, epsilon=1e-6)
    print("\tBaseline PWCCA similarity: ", pwcca_baseline, 
          "\n\tPWCCA similarity: ", pwcca_mean)


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

cka_calculator = CKA_np()

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
            calculate_cca(acts1, acts2, i)
            cal_cca_torch(acts1, acts2)
            cka_similarity = cka_calculator.cka(acts1, acts1)
            print("CKA similarity:", cka_similarity)
            

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





