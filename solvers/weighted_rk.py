import sys
import numpy as np

epsilon = 1e-5


def weighted_randomized_kaczmarz(A, b, x0=None, p=50, max_iter=10000):
    m, n = A.shape

    # Ensure that A has full rank
    if np.linalg.matrix_rank(A) < n:
        raise ValueError("Matrix A does not have full rank.")

    # Initial solution
    if x0 is None:
        x0 = np.zeros(n)
    xk = x0

    # Pre-computation
    r0 = A @ xk - b
    Q = A @ A.T

    # Iteration
    for _ in range(max_iter):
        Ax_minus_b_norm = np.linalg.norm(A @ xk - b, ord=p) ** p

        # Compute probabilities for row selection
        probabilities = np.abs(A @ xk - b) ** p / (Ax_minus_b_norm + epsilon)

        probabilities /= np.sum(probabilities)
        probabilities = np.nan_to_num(probabilities)
        i = np.random.choice(m, p=probabilities)

        # Compute lambda for the selected row
        lambda_ = (b[i] - np.dot(A[i], xk)) / Q[i, i]

        # Update xk and rk
        xk = xk + lambda_ * A[i]
        r0 = r0 + lambda_ * Q[:, i]

        # Convergence check (optional, depending on specific use-case)
        if np.linalg.norm(A @ xk - b) < 1e-6:
            break

    return xk


def standard_kaczmarz(A, b, iterations=1000):
    n = A.shape[1]
    x = np.zeros(n)
    count = 0
    while count < iterations:
        for i in range(len(b)):
            a_i = A[i]
            x = x + ((b[i] - np.dot(a_i, x)) / np.linalg.norm(a_i) ** 2) * a_i
            count += 1
    return x


# Create a synthetic linear system
# np.random.seed(42)  # For reproducibility
m, n = 100, 100  # Dimensions

# top left corner: diagonal matrix
A_dia_val = np.random.randn(n // 2)
A_rel = np.diag(A_dia_val)

# C & C_T: tridiagonal matrix
C_dia_val = np.random.randn(n // 2)
C_above_dia = np.random.randn(n // 2 - 1)
C_below_dia = np.random.randn(n // 2 - 1)
C = np.diag(C_dia_val)

for i in range(n // 2 - 1):
    C[i, i + 1] = C_above_dia[i]
for i in range(1, n // 2):
    C[i, i - 1] = C_below_dia[i - 1]

#
C_T = np.transpose(C)
A = np.zeros((m, n))
A[: m // 2, : n // 2] = A_rel
A[m // 2 :, : n // 2] = C_T
A[: m // 2, n // 2 :] = C

x_true = np.random.randn(n)
b = A.dot(x_true)

# Run both algorithms
iterations = 30000
x_standard = standard_kaczmarz(A, b, iterations=iterations)
x_random = weighted_randomized_kaczmarz(A, b, max_iter=iterations)


# Evaluate performance
def evaluate(x_pred, x_true):
    error = np.linalg.norm(x_pred - x_true)
    return error


error_standard = evaluate(x_standard, x_true)
error_random = evaluate(x_random, x_true)


# Compare weighted random Kaczmarz to standard Kaczamarz
print(f"Standard Kaczmarz Error: {error_standard}")
print(f"Weighted Random Kaczmarz Error: {error_random}")
print(np.linalg.norm(A, ord=2) ** 2)
