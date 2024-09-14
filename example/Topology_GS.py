import numpy as np
import gs
# X = np.random.rand(4096, 6)
# rlt = gs.rlts(X, L_0=32, gamma=1.0/8, i_max=100, n=100)
# mrlt = np.mean(rlt, axis=0)
# print(mrlt)

# X = np.random.rand(4096, 6)
# rltx = gs.rlts(X, n=100, L_0=32, i_max=10, gamma=1.0/8)
# Y = np.random.rand(4096, 6)
# rlty = gs.rlts(Y, n=100, L_0=32, i_max=10, gamma=1.0/8)

# score = gs.geom_score(rltx, rlty)
# print(score)


circle = gs.circle()
print(circle)
print(circle.shape)

circle = gs.filled_circle()
print(circle)
print(circle.shape)