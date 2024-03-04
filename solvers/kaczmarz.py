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

def line_search_wolfe_conditions(f, grad_f, x, d, alpha_init=0.5, c1=1e-4, c2=0.9):
    """
    Perform a line search to find the alpha that satisfies the Wolfe conditions.

    Parameters:
    - f: The objective function.
    - grad_f: The gradient of the objective function.
    - x: Current point in the iteration.
    - d: The search direction.
    - alpha_init: Initial guess for alpha.
    - c1, c2: Constants for the Wolfe conditions.

    Returns:
    - alpha: Step size that satisfies the Wolfe conditions.
    """
    alpha = alpha_init
    while True:
        new_x = x + alpha * d
        if np.all(f(new_x) <= f(x) + c1 * alpha * np.dot(grad_f(x), d)) and np.all(np.dot(grad_f(new_x), d) >= c2 * np.dot(grad_f(x), d)):
            break
        alpha *= 0.5
    return alpha

def mgrk_with_adaptive_alpha(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    beta: float,
    theta: float,
    x0: typing.Optional[np.ndarray] = None,
    max_iter=10000,
    tol=1e-6
) -> np.ndarray:
    
    if x0 is None:
        x0 = np.zeros_like(A.shape[1])
    
    x = x0.copy()
    x_prev = x0.copy()
    for _ in range(max_iter):
        # Compute the residuals and determine the set Sk
        residuals = np.abs(np.dot(A, x) - b)
        # Simplified computation for gamma_k
        gamma_k = np.linalg.norm(A, ord='fro')**2
        criterion = theta * \
            np.max(residuals)**2 + (1 - theta) * \
            np.linalg.norm(residuals)**2 / gamma_k
        Sk = np.where(residuals**2 >= criterion)[0]

        if len(Sk) == 0:
            break  # All residuals are below the threshold

        # Select ik from Sk based on some probability criterion (uniformly for simplicity)
        ik = np.random.choice(Sk)

        # Compute the search direction d
        a_ik = A[ik, :]
        numerator = np.dot(a_ik, x) - b[ik]
        denominator = np.linalg.norm(a_ik)**2
        d = -numerator / denominator * a_ik + beta * (x - x_prev)
        
        # Define f and grad_f for the line search
        f = lambda x: 0.5 * np.linalg.norm(np.dot(A, x) - b)**2
        grad_f = lambda x: np.dot(A.T, np.dot(A, x) - b)
        
        # Adaptive alpha using line search that satisfies Wolfe conditions
        alpha = line_search_wolfe_conditions(f, grad_f, x, d)

        # Update x with the found alpha
        x_next = x + alpha * d

        if np.linalg.norm(x_next - x) < tol:
            break  # Convergence criterion met

        x_prev = x
        x = x_next

    return x


if __name__ == "__main__":
    A = np.array(
        [
            [2, 1, 0, 0, 0],
            [1, 2, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 2, 1],
            [0, 0, 0, 1, 2],
        ]
    )
    b = np.array([1, 1, 1, 1, 1])
    # x = kaczmarz(A, b)

    x = mgrk_with_adaptive_alpha(A, b, 0.5, 0.5, 1)

    print(f"A: {A}\nb: {b}")
    print(f"x: {x}")
    print(f"approx. b: {np.dot(A, x)}")
    print(f'numpy x: {np.linalg.solve(A, b)}')
