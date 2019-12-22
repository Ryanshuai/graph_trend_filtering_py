import numpy as np
from numpy.linalg import norm
import prox_tv as ptv

from grid_system import grid_system_2d, grid_system_3d
from get_Delta_grid import get_Delta_grid
from soft_thresh import soft_thresh
from construct_O import construct_O


def gtf_admm_grid_v1(y: np.array, k, lamb, rho, max_iter=1000):
    y_size = y.size
    y_shape = y.shape
    y_dim = y.ndim
    if y_dim == 2:
        D = get_Delta_grid(y_shape, 'gtf2d', 0)
    elif y_dim == 3:
        D = get_Delta_grid(y_shape, 'gtf3d', 0)
    else:
        raise AssertionError('Grids with dimension  > 3 not supported')

    O = construct_O(D, k)
    if k % 2 == 0:
        O = O[:O.shape[1], :]

    tol_abs = 1e-5
    tol_rel = 1e-4

    y = y.reshape((y_size, 1), order='F')
    x = y.copy()
    z = np.zeros_like(y, dtype=np.float64)
    u = z.copy()

    for i in range(max_iter):
        if y_dim == 2:
            b = (O.T @ (rho * z - u) + y).reshape(y_shape, order='F')
            x = grid_system_2d(b, k + 1, rho)
        elif y_dim == 3:
            b = (O.T @ (rho * z - u) + y).reshape(y_shape, order='F')
            x = grid_system_3d(b, k + 1, rho)

        x = x.reshape((y_size, 1), order='F')
        Ox = O @ x
        z_new = soft_thresh(Ox + u / rho, lamb / rho)
        s = rho * norm(O.T @ (z_new - z))
        z = z_new

        u += rho * (Ox - z)
        r = norm(Ox - z)

        eps_pri = np.sqrt(y.size) * tol_abs + tol_rel * max(norm(Ox), norm(z))
        eps_dual = np.sqrt(y.size) * tol_abs + tol_rel * norm(O.T @ u)

        if i % 1 == 0:
            print('{} [r, s]={}, {}, [eps_pri, eps_dual]={},{}'.format(i, r, s, eps_pri, eps_dual))

        tau = 2
        if r > 10 * s:
            rho *= tau
        elif s > 10 * s:
            rho /= tau

        if r < eps_pri and s < eps_dual:
            print('converged.')
            break
    else:  # no break
        print('Reached maxiter.')
    return x.reshape(y_shape, order='F')
