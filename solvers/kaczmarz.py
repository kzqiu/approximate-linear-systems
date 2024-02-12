import numpy as np
import random

def kaczmarz(A: np.ndarray, b: np.ndarray, x_init: np.ndarray=None, k: int=100, scale: float=1.0):
    '''Performs randomized Kaczmarz method on linear system Ax=b.

    A: m x n matrix, where m >= n.
    x_init: 1 x m vector, arbitrary approximation of x.
    b: 1 x m vector.
    k: k >= 0, number of iterations to run kaczmarz.
    '''
    if x_init is None:
        x = np.zeros_like(A.shape[1])
    else:
        x = x_init

    for iters in range(k):
        probs = scale * np.random.random(len(b))
        for i in range(len(b)):
            if probs[i] > np.exp(-i):
                x = x + ((b[i] - np.dot(A[i], x)) / np.linalg.norm(A[i]) ** 2) * A[i]

    return x


if __name__ == "__main__":
    A = np.array([[1, 0, 0 ,0 ,0],
                 [0, 0, 1, 0, 0],
                 [1, -7, 0, 4, 2],
                 [0, 4, 2, -7, 1],
                 [0, 2, 0, 1, -7]])
    b = np.array([3, 5, 0, 0, 6])
    x = kaczmarz(A, b)

    print(f'A: {A}\nb: {b}')
    print(f'x: {x}')
    print(f'approx. b: {np.dot(A, b)}')
    # print(f'numpy x: {np.linalg.solve(A, b)}')
