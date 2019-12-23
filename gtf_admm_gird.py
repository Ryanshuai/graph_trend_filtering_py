import numpy as np
from numpy.linalg import norm
import prox_tv as ptv

from grid_system import grid_system_2d, grid_system_3d
from get_Delta_grid import get_Delta_grid
from soft_thresh import soft_thresh


def gtf_admm_grid(y: np.array, k, lamb, rho, max_iter=1000):
    if isinstance(lamb, tuple):
        lamb_x, lamb_y, lamb_z = lamb
    else:
        lamb_x, lamb_y, lamb_z = lamb, lamb, lamb

    y_size = y.size
    y_shape = y.shape
    y_dim = y.ndim
    if y_dim == 2:
        D = get_Delta_grid(y_shape, 'gtf2d', 0)
    elif y_dim == 3:
        D = get_Delta_grid(y_shape, 'gtf3d', 0)
    else:
        raise AssertionError('Grids with dimension  > 3 not supported')

    if k == 0:
        print('k=0: Direct Solution.\n')
        x = ptv.tvgen(y, [lamb_x, lamb_y, lamb_z], [1, 2, 3], [1, 1, 1])
        return x, 0

    L = D.T @ D
    Lk = np.eye(y.size)
    for i in range(k // 2):
        Lk = L @ Lk

    tol_abs = 1e-5
    tol_rel = 1e-4

    y = y.reshape((y_size, 1), order='F')
    x = y.copy()
    z = np.zeros_like(y, dtype=np.float64)
    u = z.copy()

    for i in range(max_iter):
        if y_dim == 2:
            b = (Lk @ (rho * z - u) + y).reshape(y_shape, order='F')
            x = grid_system_2d(b.reshape(y_shape), np.ceil(k / 2) * 2, rho)
        elif y_dim == 3:
            b = (Lk @ (rho * z - u) + y).reshape(y_shape, order='F')
            x = grid_system_3d(b.reshape(y_shape), np.ceil(k / 2) * 2, rho)

        x = x.reshape((y_size, 1), order='F')
        Lkx = Lk @ x

        if k % 2 == 0:
            nLkx = np.reshape((Lkx + u / rho), y_shape, order='F')
            z_new = ptv.tvgen(nLkx, [lamb_x, lamb_y, lamb_z], [1, 2, 3], [1, 1, 1])
            z_new = z_new.reshape((y_size, 1), order='F')
        else:
            z_new = soft_thresh(Lkx + u / rho, lamb / rho)

        s = rho * norm(Lk * (z_new - z))
        z = z_new

        u += rho * (Lkx - z)
        r = norm(Lkx - z)

        eps_pri = np.sqrt(y.size) * tol_abs + tol_rel * max(norm(Lkx), norm(z))
        eps_dual = np.sqrt(y.size) * tol_abs + tol_rel * norm(Lk.T @ u)

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
