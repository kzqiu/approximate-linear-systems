import numpy as np
import typing
import random


def kaczmarz(
    A: np.ndarray,
    b: np.ndarray,
    x_init: typing.Optional[np.ndarray] = None,
    k: int = 100,
    scale: float = 1.0,
) -> np.ndarray :
    """
    Performs randomized Kaczmarz method on linear system Ax=b.

    parameters:
    A -- m x n matrix, where m >= n.
    x_init -- 1 x m vector, arbitrary approximation of x.
    b -- 1 x m vector.
    k -- k >= 0, number of iterations to run kaczmarz.
    """
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

def mgrk(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    beta: float,
    theta: float,
    k: int=1,
    max_iter: int=1000
) -> np.ndarray :
    """
    Implementation of mGRK from https://arxiv.org/pdf/2307.01988.pdf
    """
    xk = np.zeros_like(b)

    for _ in range(max_iter):
        ax_minus_b_sqr = (np.dot(A, xk) - b) ** 2
        nk = np.argwhere(ax_minus_b_sqr).T.squeeze()
        gk = np.trace(np.dot(A[nk], A[nk].T))
        update = ax_minus_b_sqr / np.diag(np.dot(A, A.T))
        max_update = np.argmax(update)

        sk = np.argwhere(
            update >= theta * max_update + (1 - theta) * np.linalg.norm(np.dot(A, xk) - b) ** 2 / gk
        ).T.squeeze()

        rk = np.where(i in sk, np.zeros_like(A[0]), )

        print(probs)

        break
        

    return xk


if __name__ == "__main__":
    A = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, -7, 0, 4, 2],
            [0, 4, 2, -7, 1],
            [0, 2, 0, 1, -7],
        ]
    )
    b = np.array([3, 5, 0, 0, 6])
    # x = kaczmarz(A, b)

    mgrk(A, b, 0.5, 0.5, 1)

    # print(f"A: {A}\nb: {b}")
    # print(f"x: {x}")
    # print(f"approx. b: {np.dot(A, b)}")
    # print(f'numpy x: {np.linalg.solve(A, b)}')
