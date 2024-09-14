import numpy as np
from msid import msid_score

np.random.seed(1)

x0 = np.random.randn(4096, 6)
x1 = np.random.randn(4096, 6) # MSID can compare two data distributions with different dimensionalities
y0 = np.random.beta(0.5, 0.5, (1000, 10))

print('x0=N(0, 1), shape=', x0.shape)
print('x1=N(0, 1), shape=', x1.shape)
print('y0=beta(0.5, 0.5), shape=', y0.shape)

print('MSID(x0, x1)', msid_score(x0, x1))
print('MSID(x0, y0)', msid_score(x0, y0))