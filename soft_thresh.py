import numpy as np


def soft_thresh(X, thresh):
    Y = np.maximum(X - thresh, 0)
    Y = Y + np.minimum(X + thresh, 0)
    return Y


if __name__ == '__main__':
    mat = np.array([1, 2, 3, 4])
    a = soft_thresh(mat, 2)
    print(a)
