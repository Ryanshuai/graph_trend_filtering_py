import numpy as np
import prox_tv as ptv

from grid_system import grid_system_2d, grid_system_3d
from get_Delta_grid import get_Delta_grid
from soft_thresh import soft_thresh
from time import time


def gtf_admm_grid(y: np.array, k, lamb, rho, max_iter):
    if isinstance(lamb, tuple):
        lamb_x, lamb_y, lamb_z = lamb
    else:
        lamb_x, lamb_y, lamb_z = lamb, lamb, lamb

    sz = y.shape
    if y.ndim == 2:
        D = get_Delta_grid(sz, 'gtf2d', 0)
    elif y.ndim == 3:
        D = get_Delta_grid(sz, 'gtf3d', 0)
        # print(D.shape)
    else:
        raise AssertionError('Grids with dimension  > 3 not supported')

    if k == 0:
        print('k=0: Direct Solution.\n')
        x = ptv.tvgen(y, [lamb_x, lamb_y, lamb_z], [1, 2, 3], [1, 1, 1])
        return x, 0

    history = list()

    n = y.size

    L = D.T @ D
    Lk = np.eye(n)
    for i in range(k // 2):
        Lk = L @ Lk

    tol_abs = 1e-5
    tol_rel = 1e-4

    conv = False
    y = y.reshape(y.size, 1)
    x = y
    m = n
    z = np.zeros((n, 1))
    u = z
    iter = 1
    while not conv:
        if len(sz) == 2:
            x = grid_system_2d((Lk @ (rho * z - u) + y).reshape(sz), np.ceil(k / 2) * 2, rho)
        elif len(sz) == 3:
            x = grid_system_3d((Lk @ (rho * z - u) + y).reshape(sz), np.ceil(k / 2) * 2, rho)

        x = x.reshape(x.size, 1)
        Lkx = Lk @ x

        if k % 2 == 0:
            nLkx = np.reshape((Lkx + u / rho), sz)
            start = time()
            z_new = ptv.tvgen(nLkx, [lamb_x, lamb_y, lamb_z], [1, 2, 3], [1, 1, 1])
            stop = time()
            # print('porx_TV need: ', stop - start)
            z_new = z_new.reshape(z_new.size, 1)
            # z_new = graphtv(Lkx + u / rho, edges1, edges2, lamb / rho)
        else:
            z_new = soft_thresh(Lkx + u / rho, lamb / rho)

        norm = np.linalg.norm
        s = rho * norm(Lk * (z_new - z))
        z = z_new

        u += rho * (Lkx - z)
        r = norm(Lkx - z)

        eps_pri = np.sqrt(n) * tol_abs + tol_rel * max(norm(Lkx), norm(z))
        eps_dual = np.sqrt(m) * tol_abs + tol_rel * norm(Lk.T @ u)

        if iter % 1 == 0:
            pass
            # print('{} [r, s]={}, {}, [eps_pri, eps_dual]={},{}'.format(iter, r, s, eps_pri, eps_dual))
        if r < eps_pri and s < eps_dual:
            conv = True
            # print('converged.')
        elif iter >= max_iter:
            conv = True
            # print('Reached maxiter.')
        # history = history.append([s, r])

        tau = 2
        if r > 10 * s:
            rho *= tau
        elif s > 10 * s:
            rho /= tau

        iter += 1

    return x
