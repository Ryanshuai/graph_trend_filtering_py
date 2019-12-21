def construct_O(D, k):
    O = D
    for i in range(k):
        if i % 2 == 0:
            O = D.T @ O
        else:
            O = D @ O
    return O

if __name__ == '__main__':
    import numpy as np

    D = np.array([[1, 9], [3, 4]])
    res = construct_O(D, 1)
    print(res)
