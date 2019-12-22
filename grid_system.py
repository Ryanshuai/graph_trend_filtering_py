import numpy as np
from numpy import kron, ones
from scipy.fftpack import dct, idct


def grid_system_2d(b: np.ndarray, k, rho):
    pass


def grid_system_3d(b: np.ndarray, k, rho):
    h, w, d = b.shape

    lambs_d = 4 * np.power(np.sin(np.pi * (np.arange(0, d)) / (2 * d)), 2)
    lambs_w = 4 * np.power(np.sin(np.pi * (np.arange(0, w)) / (2 * w)), 2)
    lambs_h = 4 * np.power(np.sin(np.pi * (np.arange(0, h)) / (2 * h)), 2)

    lambs = kron(kron(ones((1, h)), ones((1, w))), lambs_d) + \
            kron(kron(ones((1, h)), lambs_w), ones((1, d))) + \
            kron(kron(lambs_h, ones((1, w))), ones((1, d)))

    lambs = lambs.reshape((h, w, d))

    sigma = rho * np.power(lambs, k) + ones((h, w, d))

    tmp = dct(b, axis=-1, norm='ortho')
    tmp = dct(tmp, axis=-2, norm='ortho')
    tmp = dct(tmp, axis=-3, norm='ortho')
    x = tmp / sigma
    x = idct(x, axis=-3, norm='ortho')
    x = idct(x, axis=-2, norm='ortho')
    x = idct(x, axis=-1, norm='ortho')

    return x


if __name__ == '__main__':
    b = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    k = 1
    rho = 1

    xx = grid_system_3d(b, k, rho)
    print(xx)