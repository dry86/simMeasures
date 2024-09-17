import numpy as np

from msid import msid_score
from gs.geom_score import *

def cal_gs(X, Y):
    rltx = rlts(X, n=100, L_0=32, i_max=10, gamma=1.0/8)

    rlty = rlts(Y, n=100, L_0=32, i_max=10, gamma=1.0/8)

    score = geom_score(rltx, rlty)
    # print(score)

    return score

def cal_msid(X, Y):

    score = msid_score(X, Y)

    return score

# 示例使用
if __name__ == "__main__":

    np.random.seed(1)

    X = np.random.rand(100, 4096)
    Y = np.random.rand(100, 4096)


    print(cal_gs(X, Y))

    print('MSID(X, Y)', cal_msid(X, Y))




