import numpy as np
from scipy.linalg import orthogonal_procrustes
A = np.array([[ 1,  0], [0,  1]])
B = np.array([[ 0,  1], [1,  0]])
R, sca = orthogonal_procrustes(A, B)

print(R)
print(sca)