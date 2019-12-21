import numpy as np
from numpy import kron, ones
from scipy.fftpack import dct, idct


def grid_system_2d(b: np.ndarray, k, rho):
    pass


def grid_system_3d(b: np.ndarray, k, rho):
    n3, n2, n1 = b.shape

    lambs1 = 4 * np.power(np.sin(np.pi * (np.arange(0, n1)) / (2 * n1)), 2)
    lambs2 = 4 * np.power(np.sin(np.pi * (np.arange(0, n2)) / (2 * n2)), 2)
    lambs3 = 4 * np.power(np.sin(np.pi * (np.arange(0, n3)) / (2 * n3)), 2)

    lambs = kron(kron(ones((1, n3)), ones((1, n2))), lambs1) + \
            kron(kron(ones((1, n3)), lambs2), ones((1, n1))) + \
            kron(kron(lambs3, ones((1, n2))), ones((1, n1)))

    lambs = lambs.reshape((n3, n2, n1))

    sigma = rho * np.power(lambs, k) + ones((n3, n2, n1))

    tmp = dct(b, axis=-1, norm='ortho')
    tmp = dct(tmp, axis=-2, norm='ortho')
    tmp = dct(tmp, axis=-3, norm='ortho')
    tmp *= 1. / sigma
    # print(tmp)
    x = idct(tmp, axis=-3, norm='ortho')
    x = idct(x, axis=-2, norm='ortho')
    x = idct(x, axis=-1, norm='ortho')

    return x


if __name__ == '__main__':
    b = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    k = 1
    rho = 1

    xx = grid_system_3d(b, k, rho)
    print(xx)