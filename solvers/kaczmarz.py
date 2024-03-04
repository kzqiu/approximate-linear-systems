import numpy as np
import typing
import random


def kaczmarz(
    A: np.ndarray,
    b: np.ndarray,
    x_init: typing.Optional[np.ndarray] = None,
    k: int = 100,
    scale: float = 1.0,
) -> np.ndarray:
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
    x0: typing.Optional[np.ndarray] = None,
    max_iter=1000,
    tol=1e-6,
) -> np.ndarray:
    """
    Solves the Ax = b system using the mGRK method from https://arxiv.org/pdf/2307.01988.pdf

    Parameters:
    - A: numpy array, the coefficient matrix A in Ax = b.
    - b: numpy array, the right-hand side vector in Ax = b.
    - alpha: float, the step size parameter.
    - beta: float, the momentum parameter.
    - theta: float, the parameter to adjust the greedy probability criterion.
    - x0: numpy array, initial guess for the solution.
    - max_iter: int, maximum number of iterations.
    - tol: float, tolerance for the stopping criterion.

    Returns:
    - x: numpy array, the approximate solution to Ax = b.
    """

    if x0 is None:
        x0 = np.zeros(A.shape[1])

    print(A.shape)
    print(x0.shape)

    x = x0.copy()
    x_prev = x0.copy()
    for _ in range(max_iter):
        # Compute the residuals and determine the set Sk
        residuals = np.abs(np.dot(A, x) - b)
        # Simplified computation for gamma_k
        gamma_k = np.linalg.norm(A, ord="fro") ** 2
        criterion = (
            theta * np.max(residuals) ** 2
            + (1 - theta) * np.linalg.norm(residuals) ** 2 / gamma_k
        )
        Sk = np.where(residuals**2 >= criterion)[0]

        if len(Sk) == 0:
            break  # All residuals are below the threshold

        # Select ik from Sk based on some probability criterion (uniformly for simplicity)
        ik = np.random.choice(Sk)

        # Update x using the mGRK formula
        a_ik = A[ik, :]
        numerator = np.dot(a_ik, x) - b[ik]
        denominator = np.linalg.norm(a_ik) ** 2

        x_next = x - (alpha * (numerator / denominator) * a_ik) + (beta * (x - x_prev))

        if np.linalg.norm(x_next - x) < tol:
            break  # Convergence criterion met

        x_prev = x
        x = x_next

    return x


if __name__ == "__main__":
    A = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, -7, 0, 4, 2],
            [0, 4, 2, -7, 1],
            [0, 2, 0, 1, -7],
        ]
    )
    b = np.array([3, 5, 0, 0, 6])
    x = mgrk(A, b, 0.75, 0.4, 0.5)

    print(f"A: {A}\nb: {b}")
    print(f"x: {x}")
    print(f"approx. b: {np.dot(A, x)}")
    print(f"numpy x: {np.linalg.solve(A, b)}")
