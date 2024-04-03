import numpy as np

class PCG:
    def __init__(self, A, b, block_size, Nblocks, guess=None, options={}):
        self.A = A
        self.b = b
        self.block_size = block_size
        self.Nblocks = Nblocks
        self.guess = guess
        if self.guess == None:
            self.guess = np.zeros(self.A.shape[0])
        self.options = options
        self.set_default_options(self.options)
        self.Pinv = self.compute_preconditioner(
            self.A, self.block_size, self.options["preconditioner_type"]
        )

    def set_default_options(self, options):
        options.setdefault("exit_tolerance", 1e-6)
        options.setdefault("max_iter", 100)
        options.setdefault("DEBUG_MODE", False)
        options.setdefault("RETURN_TRACE", False)
        options.setdefault("preconditioner_type", "BJ")
        options.setdefault("use_RK", False)
        self.validate_precon_type(options["preconditioner_type"])

    def update_A(self, A):
        self.A = A

    def update_b(self, b):
        self.b = b

    def update_guess(self, guess):
        self.guess = guess

    def update_exit_tolerance(self, tol):
        self.options["exit_tolerance"] = tol

    def update_max_iter(self, max_iter):
        self.options["max_iter"] = max_iter

    def update_preconditioner_type(self, type):
        self.validate_precon_type(type)
        self.options["preconditioner_type"] = type

    def update_DEBUG_MODE(self, mode):
        self.options["DEBUG_MODE"] = mode

    def update_RETURN_TRACE(self, mode):
        self.options["RETURN_TRACE"] = mode

    def validate_precon_type(self, precon_type):
        if not (precon_type in ["0", "J", "BJ", "SS", "SN"]):
            print(
                "Invalid preconditioner options are [0: none, J : Jacobi, BJ: Block-Jacobi, SS: Symmetric Stair]"
            )
            exit()

    def invert_matrix(self, A):
        try:
            return np.linalg.inv(A)
        except:
            if self.options.get("DEBUG_MODE"):
                print("Warning singular matrix -- using Psuedo Inverse.")
            return np.linalg.pinv(A)

    def line_search_wolfe_conditions(
        self, f, grad_f, x, d, alpha_init=0.5, c1=1e-4, c2=0.9, max_iter=20, tol=1e-10
    ):
        """
        Perform a line search to find the alpha that satisfies the Wolfe conditions with improvements.

        Parameters:
        - f: The objective function.
        - grad_f: The gradient of the objective function.
        - x: Current point in the iteration.
        - d: The search direction.
        - alpha_init: Initial guess for alpha.
        - c1, c2: Constants for the Wolfe conditions.
        - max_iter: Maximum number of iterations to prevent infinite loops.
        - tol: Tolerance for considering Wolfe conditions met, addressing numerical precision issues.

        Returns:
        - alpha: Step size that satisfies the Wolfe conditions or comes within a tolerance of satisfying.
        """
        alpha = alpha_init
        iter_count = 0  # Initialize iteration counter

        while iter_count < max_iter:
            new_x = x + alpha * d
            f_new_x = f(new_x)  # Evaluate f at new_x once per iteration
            grad_f_new_x = grad_f(new_x)  # Evaluate grad_f at new_x once per iteration

            # Check Wolfe conditions with added tolerance
            wolfe_1 = f_new_x <= f(x) + c1 * alpha * np.dot(grad_f(x), d) + tol
            wolfe_2 = np.dot(grad_f_new_x, d) >= c2 * np.dot(grad_f(x), d) - tol

            if wolfe_1 and wolfe_2:
                break  # Wolfe conditions satisfied

            alpha *= 0.5  # Halve alpha for the next iteration
            iter_count += 1  # Increment iteration counter

        # Return the final alpha, even if Wolfe conditions are not perfectly met to avoid infinite loops
        return alpha

    def mgrk_with_adaptive_alpha(
        self,
        A: np.ndarray,
        b: np.ndarray,
        pinv: np.ndarray,
        alpha: float,
        beta: float,
        theta: float,
        x0=None,
        max_iter=10000,
        tol=1e-6,
        exit_rows=-1,
        lambduhh=1.,
        weight_exit=False,
    ) -> np.ndarray:
        A_tilde = np.dot(pinv, A)

        dim = A.shape[1]

        # if x0 is None:
        x0 = np.zeros(dim)

        if exit_rows <= 0:
            exit_rows = dim

        x = x0.copy()
        x_prev = x0.copy()
        # print("x0: " + str(x0))
        b_tilde = np.dot(pinv, b)
        b_tilde.shape[1]
        b_tilde = b_tilde.T.squeeze()
        # print("b_tilde: " + str(b_tilde))
        for _ in range(max_iter):
            # Step 1
            # Compute the residuals
            residuals = np.abs(np.dot(A_tilde, x) - b_tilde)
            # print("Residuals: " + str(residuals))
            N_k = np.where(residuals != 0)[0]
            # print("N_k: " + str(N_k))
            
            gamma_k = np.sum(np.linalg.norm(A_tilde[i], ord=2)**2 for i in N_k)
            # print("Gamma_k: " + str(gamma_k))
            
            # Step 2
            residual_divided = ((residuals[N_k]) ** 2)/(np.linalg.norm(A_tilde[N_k], ord=2, axis=1) ** 2)
            # if _ == 0:
            #     print("Residuals: " + str((residuals) ** 2))
            #     print("A_tilde Norm: " + str((np.linalg.norm(A_tilde, ord=2, axis=1) ** 2)))
            #     print("Shape of A_tilde Norm: " + str(((np.linalg.norm(A_tilde, ord=2, axis=1) ** 2)).shape))
            #     print("Shape of residuals: " + str(((residuals) ** 2).shape))
            # print("A_tilde Norm: " + str((np.linalg.norm(A_tilde, ord=2, axis=1) ** 2)))
            # print(np.linalg.norm(A_tilde[0], ord=2)**2)
            # print("ai_norm: " + str(ai_norm))
            # print(theta * np.max(((residuals) ** 2)/ai_norm))
            criterion = (
                theta * np.max(residual_divided)
                + (1 - theta) * ((np.linalg.norm(np.dot(A_tilde,x)-b_tilde, ord=2) ** 2) / gamma_k)
            )
            # print("Gamma side: " + str((1 - theta) * (np.linalg.norm(np.dot(A_tilde,x)-b_tilde, ord=2) ** 2 / gamma_k)))
            # print("Residuals divided: " + str(residual_divided))
            # print("Criterion: " + str(criterion))
            Sk = np.where(residual_divided >= criterion)[0]

            # print(Sk)
            if len(Sk) == 0:
                break  # All residuals are below the threshold

            # Select ik from Sk based on some probability criterion (uniformly for simplicity)
            ik = np.random.choice(Sk)

            # Compute the search direction d
            a_ik = A_tilde[ik, :]
            numerator = np.dot(a_ik, x) - b_tilde[ik]
            denominator = np.linalg.norm(a_ik) ** 2
            direction = (-numerator / denominator * a_ik)
            alpha_direction = (-numerator / denominator * a_ik) + (beta * (x - x_prev))

            # Define f and grad_f for the line search
            f = lambda x: 0.5 * np.linalg.norm(np.dot(A_tilde, x) - b_tilde) ** 2
            grad_f = lambda x: np.dot(A_tilde.T, np.dot(A_tilde, x) - b_tilde)

            # Adaptive alpha using line search that satisfies Wolfe conditions
            # alpha = self.line_search_wolfe_conditions(f, grad_f, x, alpha_direction)

            # Update x with the found alpha
            x_next = x + alpha * direction + (beta * (x - x_prev))

            exit_vec = (x_next - x)[:min(dim, exit_rows)]

            if weight_exit:
                exit_vec = exit_vec * np.exp(-lambduhh * np.arange(exit_rows))

            if np.linalg.norm(exit_vec) < tol:
                break  # Convergence criterion met
            
            
            # exit_norm = np.linalg.norm(np.dot(A_tilde,x) - b_tilde) ** 2
            # if exit_norm < tol:
            #     break

            x_prev = x
            x = x_next

        print(_)

        x = x.reshape(len(x), 1)

        # print("x: " + str(x))
        return x

    def pcg(self, A, b, Pinv, guess, options={}):
        self.set_default_options(options)
        trace = []

        if options["use_RK"]:
            # return self.prk(A, b, Pinv, guess, options)
            # return self.weighted_randomized_kaczmarz(A, b, Pinv)
            # return self.mgrk(A, b, Pinv, 0.6, 0.5, 0.8, exit_rows=10, weight_exit=True)
            # print("b: " + str(b))
            return self.mgrk_with_adaptive_alpha(
                A, b, Pinv, 0.6, 0.5, .9, guess, lambduhh=0.1, weight_exit=True
            )

        # if options['only_precon']:
        # return np.matmul(Pinv,b)

        # initialize
        x = np.reshape(guess, (guess.shape[0], 1))
        r = b - np.matmul(A, x)

        r_tilde = np.matmul(Pinv, r)
        p = r_tilde
        nu = np.matmul(r.transpose(), r_tilde)
        if options["DEBUG_MODE"]:
            print("Initial nu[", nu, "]")
        if options["RETURN_TRACE"]:
            trace = nu[0].tolist()
            trace2 = [np.linalg.norm(b - np.matmul(A, x))]
        # loop
        for iteration in range(options["max_iter"]):
            Ap = np.matmul(A, p)
            alpha = nu / np.matmul(p.transpose(), Ap)
            r -= alpha * Ap
            x += alpha * p

            r_tilde = np.matmul(Pinv, r)
            nu_prime = np.matmul(r.transpose(), r_tilde)
            if options["RETURN_TRACE"]:
                trace.append(nu_prime.tolist()[0][0])
                trace2.append(np.linalg.norm(b - np.matmul(A, x)))

            if abs(nu_prime) < options["exit_tolerance"]:
                if options["DEBUG_MODE"]:
                    print(Pinv)
                    print("Exiting with err[", abs(nu_prime), "]")
                break
            else:
                if options["DEBUG_MODE"]:
                    print("Iter[", iteration, "] with err[", abs(nu_prime), "]")

            beta = nu_prime / nu
            p = r_tilde + beta * p
            nu = nu_prime

        # print(iteration)
        if options["RETURN_TRACE"]:
            trace = list(map(abs, trace))
            return x, (trace, trace2)
        else:
            return x

    def compute_preconditioner(self, A, block_size, preconditioner_type):
        if preconditioner_type == "0":  # null aka identity
            return np.identity(A.shape[0])

        if preconditioner_type == "J":  # Jacobi aka Diagonal
            return self.invert_matrix(np.diag(np.diag(A)))

        elif preconditioner_type == "BJ":  # Block-Jacobi
            n_blocks = int(A.shape[0] / block_size)
            Pinv = np.zeros(A.shape)
            for k in range(n_blocks):
                rc_k = k * block_size
                rc_kp1 = rc_k + block_size
                Pinv[rc_k:rc_kp1, rc_k:rc_kp1] = self.invert_matrix(
                    A[rc_k:rc_kp1, rc_k:rc_kp1]
                )

            return Pinv

        elif (
            preconditioner_type == "SS"
        ):  # Symmetric Stair (for blocktridiagonal of blocksize nq+nv)
            n_blocks = int(A.shape[0] / block_size)
            Pinv = np.zeros(A.shape)
            # compute stair inverse
            for k in range(n_blocks):
                # compute the diagonal term
                Pinv[
                    k * block_size : (k + 1) * block_size,
                    k * block_size : (k + 1) * block_size,
                ] = self.invert_matrix(
                    A[
                        k * block_size : (k + 1) * block_size,
                        k * block_size : (k + 1) * block_size,
                    ]
                )
                if np.mod(k, 2):  # odd block includes off diag terms
                    # Pinv_left_of_diag_k = -Pinv_diag_k * A_left_of_diag_k * -Pinv_diag_km1
                    Pinv[
                        k * block_size : (k + 1) * block_size,
                        (k - 1) * block_size : k * block_size,
                    ] = -np.matmul(
                        Pinv[
                            k * block_size : (k + 1) * block_size,
                            k * block_size : (k + 1) * block_size,
                        ],
                        np.matmul(
                            A[
                                k * block_size : (k + 1) * block_size,
                                (k - 1) * block_size : k * block_size,
                            ],
                            Pinv[
                                (k - 1) * block_size : k * block_size,
                                (k - 1) * block_size : k * block_size,
                            ],
                        ),
                    )
                elif (
                    k > 0
                ):  # compute the off diag term for previous odd block (if it exists)
                    # Pinv_right_of_diag_km1 = -Pinv_diag_km1 * A_right_of_diag_km1 * -Pinv_diag_k
                    Pinv[
                        (k - 1) * block_size : k * block_size,
                        k * block_size : (k + 1) * block_size,
                    ] = -np.matmul(
                        Pinv[
                            (k - 1) * block_size : k * block_size,
                            (k - 1) * block_size : k * block_size,
                        ],
                        np.matmul(
                            A[
                                (k - 1) * block_size : k * block_size,
                                k * block_size : (k + 1) * block_size,
                            ],
                            Pinv[
                                k * block_size : (k + 1) * block_size,
                                k * block_size : (k + 1) * block_size,
                            ],
                        ),
                    )
            # make symmetric
            for k in range(n_blocks):
                if np.mod(k, 2):  # copy from odd blocks
                    # always copy up the left to previous right
                    Pinv[
                        (k - 1) * block_size : k * block_size,
                        k * block_size : (k + 1) * block_size,
                    ] = Pinv[
                        k * block_size : (k + 1) * block_size,
                        (k - 1) * block_size : k * block_size,
                    ].transpose()
                    # if not last block copy right to next left
                    if k < n_blocks - 1:
                        Pinv[
                            (k + 1) * block_size : (k + 2) * block_size,
                            k * block_size : (k + 1) * block_size,
                        ] = Pinv[
                            k * block_size : (k + 1) * block_size,
                            (k + 1) * block_size : (k + 2) * block_size,
                        ].transpose()
            return Pinv

        elif (
            preconditioner_type == "SN"
        ):  # [0] == "S" and preconditioner_type[1:].isnumeric(): # Stair to the Nth (for blocktridiagonal of blocksize nq+nv)
            series_levels = 100  # int(preconditioner_type[1:])
            n_blocks = int(A.shape[0] / block_size)
            Pinv = np.zeros(A.shape)
            Q = np.zeros(A.shape)
            # compute stair inverse
            for k in range(n_blocks):
                # compute the diagonal term
                Pinv[
                    k * block_size : (k + 1) * block_size,
                    k * block_size : (k + 1) * block_size,
                ] = self.invert_matrix(
                    A[
                        k * block_size : (k + 1) * block_size,
                        k * block_size : (k + 1) * block_size,
                    ]
                )
                if np.mod(k, 2):  # odd block includes off diag terms
                    # Pinv_left_of_diag_k = -Pinv_diag_k * A_left_of_diag_k * -Pinv_diag_km1
                    Pinv[
                        k * block_size : (k + 1) * block_size,
                        (k - 1) * block_size : k * block_size,
                    ] = -np.matmul(
                        Pinv[
                            k * block_size : (k + 1) * block_size,
                            k * block_size : (k + 1) * block_size,
                        ],
                        np.matmul(
                            A[
                                k * block_size : (k + 1) * block_size,
                                (k - 1) * block_size : k * block_size,
                            ],
                            Pinv[
                                (k - 1) * block_size : k * block_size,
                                (k - 1) * block_size : k * block_size,
                            ],
                        ),
                    )
                elif (
                    k > 0
                ):  # compute the off diag term for previous odd block (if it exists)
                    # Pinv_right_of_diag_km1 = -Pinv_diag_km1 * A_right_of_diag_km1 * -Pinv_diag_k
                    Pinv[
                        (k - 1) * block_size : k * block_size,
                        k * block_size : (k + 1) * block_size,
                    ] = -np.matmul(
                        Pinv[
                            (k - 1) * block_size : k * block_size,
                            (k - 1) * block_size : k * block_size,
                        ],
                        np.matmul(
                            A[
                                (k - 1) * block_size : k * block_size,
                                k * block_size : (k + 1) * block_size,
                            ],
                            Pinv[
                                k * block_size : (k + 1) * block_size,
                                k * block_size : (k + 1) * block_size,
                            ],
                        ),
                    )

            # form the remainder
            for k in range(n_blocks):
                # start to the right then down -- so only exists for even block-row and block-column minus and plus 1
                if (k % 2) == 0:
                    start = k * block_size
                    if k > 0:  # block-column minus 1 exists
                        Q[start : start + block_size, start - block_size : start] = -A[
                            start : start + block_size, start - block_size : start
                        ]
                    if k < n_blocks - 1:  # block-column plus 1 exists
                        Q[
                            start : start + block_size,
                            start + block_size : start + 2 * block_size,
                        ] = -A[
                            start : start + block_size,
                            start + block_size : start + 2 * block_size,
                        ]
            # compute the series Final_Pinv = (SUM H^k) * Pinv

            H = np.matmul(Pinv, Q)
            sumterm = np.eye(A.shape[0])
            base = np.eye(A.shape[0])

            # Ainv = np.linalg.inv(A)
            # err = Ainv - np.matmul(sumterm,Pinv)
            # print("At level 0 err is ", np.linalg.norm(err))

            for series_level in range(series_levels):
                base = np.matmul(base, H)
                sumterm += base
                # err = Ainv - np.matmul(sumterm,Pinv)
                # print("At level X err is ", np.linalg.norm(err))
            res = np.matmul(sumterm, Pinv)

            # err = Ainv - res
            # print("At end err is ", np.linalg.norm(err))
            return (res.T + res) / 2

    def solve(self):
        return self.pcg(self.A, self.b, self.Pinv, self.guess, self.options)