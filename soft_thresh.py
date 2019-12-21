import numpy as np


def soft_thresh(X, thresh):
    Y = np.max(X - thresh, 0)
    Y = Y + np.min(X + thresh, 0)
    return Y
