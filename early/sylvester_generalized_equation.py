""""
Implementation of A note on the numerical approximate solutions for generalized
Sylvester matrix equations with applications (Bouhamidi and Jbilou, 2008)
"""
from __future__ import division
from functools import partial
import time
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state
from sklearn.linear_model import LinearRegression


def _inner_product(A, B):
    return np.trace(A.T.dot(B))


def _compute_residual(V, H, R_norm):
    n, p = V[0].shape
    q, _ = np.linalg.qr(H)
    gamma = R_norm * q[0, 0]
    R_norm_new = np.abs(gamma)
    R_k = gamma * np.concatenate(V, axis=1).dot(np.kron(q.T[-1, :],
                                                        np.eye(p, p)).T)
    return R_k, R_norm_new


def _M_T(A, B, X):
    """
    computes R = sum(A.T[q]XB.T[q])

    A: List of length q
        List of matrices nxn
    B: list of length q
        list of matrices pxp
    X: array-like
        matrix nxp

    :return: array-like of shape nxp
    """
    if len(A) != len(B):
        raise ValueError("Different length of the input matrices")

    matrix = np.zeros_like(X)
    for q in range(len(A)):
        matrix += np.linalg.multi_dot((A[q].T, X, B[q].T))
    return matrix


def _M(A, B, X):
    """
    computes R = sum(A[q]XB[q])

    A: List of length q
        List of matrices nxn
    B: list of length q
        list of matrices pxp
    X: array-like
        matrix nxp

    :return: array-like of shape nxp
    """
    if len(A) != len(B):
        raise ValueError("Different length of the input matrices")

    matrix = np.zeros_like(X)
    for q in range(len(A)):
        #print(A[q].dot(X.dot(B[q])))
        matrix += A[q].dot(X.dot(B[q]))
    return matrix


def global_arnoldi_algorithm(M, V, n_iter=100, verbose=0):
    """
    M: callable
        M(X) -> sum(AXB)
    V: array-like (nxp)
        Initial matrix to compute the span

    Returns
    V: array-like of shape ((nxp)xk)
        The final orthonormal basis
    H: array-like of shape ((k+1)xk)
        Hessenber matrix
    k: int
        The number of orthonormal vectors
    """

    H = np.zeros((n_iter+1, n_iter))  # maximum possible size
    Vs = [V]  # [V/np.linalg.norm(V)]
    for j in range(n_iter):
        V_tilde = M(X=Vs[j])
        for i in range(j):
            H[i, j] = _inner_product(Vs[i], V_tilde)
            V_tilde -= H[i, j]*Vs[i]
        H[j+1, j] = np.linalg.norm(V_tilde)
        if np.all(np.isclose(V_tilde, np.zeros_like(V_tilde))):
            break
        V_tilde = V_tilde/H[j+1, j]

        Vs.append(V_tilde)
    else:
        if verbose:
            print("The algorithm did not converge, there are more vectors than"
                  "%d", n_iter)
    if(j == n_iter-1):
        j = j+1
        H = H[:j+2, :j+1]
    else:
        H = H[:j+1, :j]

    return np.array(Vs), H, j


def _GIGMRES(A, B, C, tol=1e-3, max_iter=200, verbose=0, random_state=0):
    n, p = C.shape
    M = partial(_M, A=A, B=B)

    X = np.zeros((n, p))
    R = C - M(X=X)
    R_norm = np.linalg.norm(R)

    for _iter in range(max_iter):
        V = R / R_norm
        Vs, H, k = global_arnoldi_algorithm(M, V, n_iter=max_iter,
                                            verbose=verbose)
        if verbose:
            print("Found %d vector in the orthonormal basis", k)
        V = np.concatenate(Vs[:-1], axis=1)
        least_square = LinearRegression(fit_intercept=False)
        output = np.zeros((k+1))

        output[0] = 1
        output *= R_norm
        least_square.fit(H, output)

        y = least_square.coef_
        kron = np.kron(y, np.eye(p, p)).T
        X += np.dot(V, kron)

        q, _ = np.linalg.qr(H)
        gamma = R_norm*q[0, :][-1]
        R_norm = np.abs(gamma)
        kron2 = np.kron(q[-1, :].T, np.eye(p, p)).T
        R = gamma*V.dot(kron2)
        # R = C - M(X=X)
        # R_norm = np.linalg.norm(R)
        if verbose:
            print("The residual at iteration %d is %f", _iter, R_norm)

        if R_norm < tol:
            break

    else:
        if verbose:
            print("The algorithm did not converge")
    return X


class GeneralizedSylvesterSolver(BaseEstimator):
    """

    mode: string, optional, default: gradient
        if gigmres optimization follows Bouhamidi and Jbilou, 2008
        if gradient optimization follows Ding et al, 2008
    """

    def __init__(self, mode='gradient', max_iter=100, tol=1e-5,
                 random_state=None, verbose=0):
        self.max_iter = max_iter
        self.mode = mode
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, A, B, C):
        """

        """
        if len(A) != len(B):
            raise ValueError("Mistmatching length in the two lists of"
                             "multiplicators")

        self.random_state = check_random_state(self.random_state)
        self.C = check_array(C)
        n, p = C.shape

        for q in range(len(A)):
            A[q] = check_array(A[q])
            B[q] = check_array(B[q])
            if A[q].shape != (n, n):
                raise ValueError("Mistmatching dimensions in matrices A")
            if B[q].shape != (p, p):
                raise ValueError("Mistmatching dimensions in matrices B")

        if str(self.mode).lower() == 'gigmres':
            self.X = _GIGMRES(A=A, B=B, C=C, tol=self.tol,
                              max_iter=self.max_iter, verbose=self.verbose,
                              random_state=self.random_state)
        elif str(self.mode).lower() == 'gradient':
            self.X = _iterative_gradient_solver(A=A, B=B, C=C, tol=self.tol,
                                                max_iter=self.max_iter,
                                                verbose=self.verbose,
                                                random_state=self.random_state)
        return self


if __name__ == "__main__":
    print("Trying with small matrices and only two elements in the sum")
    A = [np.random.rand(3, 3), np.random.rand(3, 3)]
    B = [np.random.rand(2, 2), np.random.rand(2, 2)]
    X = np.random.rand(3, 2)
    C = _M(A=A, B=B, X=X)
    t1 = time.time()
    gss = GeneralizedSylvesterSolver(mode='gigmres', verbose=1, max_iter=100,
                                     tol=1e-5)
    gss.fit(A, B, C)
    print("Time %f" % (time.time() - t1))
    print(X)
    print(gss.X)
    assert(np.all(np.isclose(gss.X, X, atol=1e-2, rtol=1e-2)))

    print("Trying with small matrices")
    A = [np.random.rand(3, 3), np.random.rand(3, 3),
         np.random.rand(3, 3), np.random.rand(3, 3),
         np.random.rand(3, 3), np.random.rand(3, 3)]
    B = [np.random.rand(2, 2), np.random.rand(2, 2),
         np.random.rand(2, 2), np.random.rand(2, 2),
         np.random.rand(2, 2), np.random.rand(2, 2)]
    X = np.random.rand(3, 2)
    C = _M(A=A, B=B, X=X)
    t1 = time.time()
    gss = GeneralizedSylvesterSolver(mode='gigmres', verbose=1, max_iter=100,
                                     tol=1e-5)
    gss.fit(A, B, C)
    print("Time %f" % (time.time() - t1))
    print(X)
    print(gss.X)
    assert(np.all(np.isclose(gss.X, X, atol=1e-2, rtol=1e-2)))

    print("Trying with small matrices")
    A = [np.random.rand(10, 10), np.random.rand(10, 10)]
    B = [np.random.rand(15, 15), np.random.rand(15, 15)]
    X = np.random.rand(10, 15)
    C = _M(A=A, B=B, X=X)
    t1 = time.time()
    gss = GeneralizedSylvesterSolver(mode='gigmres', verbose=1, max_iter=500,
                                     tol=1e-5)
    gss.fit(A, B, C)
    print(X[:5,:5])
    print(gss.X[:5,:5])
    print("Time %f" % (time.time() - t1))

    print("Trying with BIG matrices")
    A = [np.random.rand(100, 100), np.random.rand(100, 100)]
    B = [np.random.rand(150, 150), np.random.rand(150, 150)]
    X = np.random.rand(1000, 1500)
    C = _M(A=A, B=B, X=X)
    t1 = time.time()
    gss = GeneralizedSylvesterSolver(mode='gigmres', verbose=1, max_iter=500,
                                     tol=1e-3)
    gss.fit(A, B, C)
    print(X[:5,:5])
    print(gss.X[:5,:5])
    print("Time %f" % (time.time() - t1))
