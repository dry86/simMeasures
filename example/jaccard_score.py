import numpy as np
from sklearn.metrics import jaccard_score
y_true = np.array([[0, 1, 1],
                   [1, 1, 0]])
y_pred = np.array([[1, 1, 1],
                   [1, 0, 0]])

print( jaccard_score(y_true[0], y_pred[0]) )

print( jaccard_score(y_true, y_pred, average="micro") )

