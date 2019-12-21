import numpy as np
from construct_O import construct_O


def get_Delta_grid(sz, method, k):
    if method == 'tf2d':
        n1, n2 = sz
        D1dx = np.diff(np.eye(n1), n=k + 1, axis=0)
        D1dy = np.diff(np.eye(n2), n=k + 1, axis=0)

        Delta1 = np.kron(np.eye(n2), D1dx)
        Delta2 = np.kron(D1dy, np.eye(n1))
        Delta = np.concatenate([Delta1, Delta2], axis=0)
        return Delta

    elif method == 'tf3d':
        n1, n2, n3 = sz
        D1dx = np.diff(np.eye(n1), n=k + 1, axis=0)
        D1dy = np.diff(np.eye(n2), n=k + 1, axis=0)
        D1dz = np.diff(np.eye(n3), n=k + 1, axis=0)

        Delta1 = np.kron(np.eye(n3), np.kron(np.eye(n2), D1dx))
        Delta2 = np.kron(np.eye(n3), np.kron(D1dy, np.eye(n1)))
        Delta3 = np.kron(D1dz, np.kron(np.eye(n2), np.eye(n1)))

        Delta = np.concatenate([Delta1, Delta2, Delta3], axis=0)
        return Delta

    elif method == 'gtf2d':
        n1, n2 = sz
        D1dx = np.diff(np.eye(n1), n=1, axis=0)
        D1dy = np.diff(np.eye(n2), n=1, axis=0)

        Delta1 = np.kron(np.eye(n2), D1dx)
        Delta2 = np.kron(D1dy, np.eye(n1))
        D1 = np.concatenate([Delta1, Delta2], axis=0)
        Delta = construct_O(D1, k)
        return Delta

    elif method == 'gtf3d':
        n1, n2, n3 = sz
        D1dx = np.diff(np.eye(n1), n=1, axis=0)
        D1dy = np.diff(np.eye(n2), n=1, axis=0)
        D1dz = np.diff(np.eye(n3), n=1, axis=0)

        Delta1 = np.kron(np.eye(n3), np.kron(np.eye(n2), D1dx))
        Delta2 = np.kron(np.eye(n3), np.kron(D1dy, np.eye(n1)))
        Delta3 = np.kron(D1dz, np.kron(np.eye(n2), np.eye(n1)))
        D1 = np.concatenate([Delta1, Delta2, Delta3], axis=0)

        Delta = construct_O(D1, k)
        return Delta

    else:
        print('Method: ', method, ' not supported.')


if __name__ == '__main__':
    delta = get_Delta_grid([3, 3], 'gtf2d', 1)
    print(delta)

    print(delta.shape)
