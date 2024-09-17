import numpy as np
from gs.geom_score import *
from gs.top_utils import circle,filled_circle

# X = np.random.rand(4096, 100)
# rltx = rlts(X, n=100, L_0=32, i_max=10, gamma=1.0/8)
# Y = np.random.rand(4096, 100)
# rlty = rlts(Y, n=100, L_0=32, i_max=10, gamma=1.0/8)

# score = geom_score(rltx, rlty)
# print(score)

# circle = circle()
# # print(circle)
# print(circle.shape)

# circle = filled_circle()
# # print(circle)
# print(circle.shape)

X = np.random.rand(100, 4096)
Y = np.random.rand(100, 4096)

def cal_gs(X, Y):
    rltx = rlts(X, n=100, L_0=32, i_max=10, gamma=1.0/8)

    rlty = rlts(Y, n=100, L_0=32, i_max=10, gamma=1.0/8)

    score = geom_score(rltx, rlty)
    print(score)

    return score

cal_gs(X, Y)

