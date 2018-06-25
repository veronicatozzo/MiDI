import warnings
from collections import defaultdict

import scipy.linalg as spla
import numpy as np

from .utils import _get_positive_negative, _par_bdot, _transpose,\
                  _compute_error


def _update_G(G, R, S, thetas_pos, thetas_neg, verbose=0):
    G_enum = {r: np.zeros(Gr.shape) for r, Gr in G.items()}
    G_denom = {r: np.zeros(Gr.shape) for r, Gr in G.items()}

    for r in R:
        i, j = r
        for l in range(len(R[r])):
            if verbose > 1:
                print("Update G due to R_%s,%s^(%d)" % (i, j, l))

            tmp1 = np.linalg.multi_dot((R[r][l], G[j, j], S[i, j][l].T))
            tmp1 = np.nan_to_num(tmp1)
            tmp1p, tmp1n = _get_positive_negative(tmp1)

            tmp2 = np.linalg.multi_dot((S[i, j][l], G[j, j].T,
                                        G[j, j], S[i, j][l].T))
            tmp2 = np.nan_to_num(tmp2)
            tmp2p, tmp2n = _get_positive_negative(tmp2)

            tmp4 = np.linalg.multi_dot((R[i, j][l].T, G[i, i], S[i, j][l]))
            tmp4 = np.nan_to_num(tmp4)
            tmp4p, tmp4n = _get_positive_negative(tmp4)

            tmp5 = np.linalg.multi_dot((S[i, j][l].T, G[i, i].T,
                                        G[i, i], S[i, j][l]))
            tmp5 = np.nan_to_num(tmp5)
            tmp5p, tmp5n = _get_positive_negative(tmp5)

            G_enum[i, i] += tmp1p + np.dot(G[i, i], tmp2n)
            G_denom[i, i] += tmp1n + np.dot(G[i, i], tmp2p)

            G_enum[j, j] += tmp4p + np.dot(G[j, j], tmp5n)
            G_denom[j, j] += tmp4n + np.dot(G[j, j], tmp5p)

            # if r == Rpr:
            #     for t in range(I.shape[1]):
            #         It = np.reshape(I[:, t], (len(I[:, t]), 1))
            #         bt = np.reshape(B[t, :], (1, len(B[t, :])))
            #         IBS = np.linalg.multi_dot((It, bt,
            #                                    S[i, j][l].T))
            #         IBS_pos, IBS_neg = _get_positive_negative(IBS)
            #         G_t = G[i, i] * I[:, t][:, np.newaxis]
            #         GSBBS = np.linalg.multi_dot((G_t, S[i, j][l], bt.T,
            #                                      bt, S[i, j][l].T))
            #         GSBBS_pos, GSBBS_neg = _get_positive_negative(GSBBS)
            #         G_enum[i, i] += IBS_pos + GSBBS_neg
            #         G_denom[i, i] += IBS_neg + GSBBS_pos

    if verbose > 1:
        print("Update of G due to constraint matrices")

    for r, thetas_p in thetas_pos.items():
        if verbose > 1:
            print("Considering Theta pos. %s" % str(r))
        for theta_p in thetas_p:
            G_denom[r, r] += np.dot(theta_p, G[r, r])

    for r, thetas_n in thetas_neg.items():
        if verbose > 1:
            print("Considering Theta neg. %s" % str(r))
        for theta_n in thetas_n:
            G_enum[r, r] += np.dot(theta_n, G[r, r])

    for r in G:
        G[r] = np.multiply(
                 G[r], np.sqrt(np.divide(G_enum[r],
                                         np.maximum(G_denom[r],
                                         np.finfo(np.float).eps))))
    return G


def _update_S(G, R, obj_types, n_jobs, verbose):
    pGtG = {}
    for r in G:
        if verbose > 1:
            print("Computing GrtGr: %s", str(r))
        GrtGr = np.nan_to_num(np.dot(G[r].T, G[r]))
        pGtG[r] = spla.pinv(GrtGr)

    if verbose > 1:
        print("Start to update S")

    tmp1 = _par_bdot(G, pGtG, obj_types, verbose, n_jobs)
    tmp2 = _par_bdot(R, tmp1, obj_types, verbose, n_jobs)
    tmp3 = _par_bdot(_transpose(G), tmp2, obj_types, verbose, n_jobs)
    S = _par_bdot(pGtG, tmp3, obj_types, verbose, n_jobs)
    return S


def _dfmf(R, G, S, thetas, types, max_iter=100, stopping=None,
          verbose=0, tol=1e-3, callback=None, n_jobs=1):
    """Data fusion by matrix factorization.

    Parameters
    ----------
    R : dictionary of array-like objects
        Relation matrices.

    ranks: dictionary of int
        The ranks of each data type.

    thetas : dictionary of array-like objects
        Constraint matrices.

    max_iter : int, optional (default=10)
        Maximum number of iterations to perform.

    init_type : string, optional (default="random_vcol")
        The algorithm to initialize latent matrix factors.

    stopping_relation : tuple (target_matrix), optional (default=None)
        Stopping criterion. Terminate iteration if the reconstruction
        error of target matrix improves by less than tol.

    verbose : int, optional (default=0)
         The amount of verbosity. Larger value indicates greater verbosity.

    callback : callable, optional
        An optional user-supplied function to call after each iteration. Called
        as callback(G, S, cur_iter), where S and G are current latent estimates

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance
        used by np.random.

    n_jobs: int (default=1)
        Number of jobs to run in parallel

    Returns
    -------
    G :
    S :
    n_iter:
    """
    err = [1e10]

    if verbose > 1:
        print("Solving for Theta_p and Theta_n")

    # getting positive and negative thetas
    thetas_pos, thetas_neg = defaultdict(list), defaultdict(list)
    for t, thetas_of_t in thetas.items():
        for theta in thetas_of_t:
            pos, neg = _get_positive_negative(theta)
            thetas_pos[t].append(pos)
            thetas_neg[t].append(neg)  # meno uno non fa casini coi segni?

    obj = []

    for _iter in range(max_iter):

        S = _update_S(G, R, types, n_jobs, verbose=0)
        G = _update_G(G, R, S, thetas_pos, thetas_neg, verbose=verbose)

        # Reconstruction error
        error = _compute_error(R, G, S, verbose)
        obj.append(error)
        if stopping:
            t1, t2 = stopping
            R_reconstructed = np.linalg.multi_dot((G[t1, t1],
                                                   S[t1, t2], G[t2, t2].T))
            diff = R[t] - R_reconstructed
            error = np.linalg.norm(diff)
        err.append(error)

        if callback:
            callback(G, S, iter)

        if verbose:
            print("Iter: %d\t obj: %5.4f\t diff_obj: %4f" %
                  (_iter, error, err[-1] - err[-2]))
        # convergence check
        if np.abs(err[-1] - err[-2]) < tol:
            break

    else:
        warnings.warn("Algorithm did not converge", Warning)

    if verbose:
        print("Violations of optimization objective: %d/%d, iter=%d" % (
                      int(np.sum(np.diff(obj) > 0)), len(obj), _iter))

    return_ = dict(G=G, S=S, n_iter=_iter)
    return return_


def _transform(R, Theta, _type, ranks, G, S, G_t, max_iter=100,
               verbose=0, tol=1e-3, random_state=None):

    obj, err = [], [1e10]
    thetas_pos, thetas_neg = defaultdict(list), defaultdict(list)
    for t, thetas_of_t in Theta.items():
        for theta in thetas_of_t:
            pos, neg = _get_positive_negative(theta)
            thetas_pos[t].append(pos)
            thetas_neg[t].append(neg)

    for _iter in range(max_iter):
        if verbose > 1:
            print("Factorization iteration: %d" % _iter)

        G_enum = np.zeros(G_t.shape)
        G_denom = np.zeros(G_t.shape)

        for r in R:
            i, j = r
            for l in range(len(R[r])):
                if verbose > 1:
                    print("Update G due to R_%s,%s^(%d)" % (i, j, l))

                if i is _type:
                    tmp1 = np.linalg.multi_dot((R[i, j][l], G[j, j],
                                                S[i, j][l].T))
                    tmp1p, tmp1n = _get_positive_negative(tmp1)
                    tmp2 = np.linalg.multi_dot((S[i, j][l], G[j, j].T, G[j, j],
                                                S[i, j][l].T))
                    tmp2p, tmp2n = _get_positive_negative(tmp2)
                    G_enum += tmp1p + np.dot(G_t, tmp2n)
                    G_denom += tmp1n + np.dot(G_t, tmp2p)

                if j is _type:
                    tmp4 = np.linalg.multi_dot(R[i, j][l].T, (G[i, i],
                                               S[i, j][l]))
                    tmp4p, tmp4n = _get_positive_negative(tmp4)
                    tmp5 = np.linalg.mulit_dot((S[i, j][l].T, G[i, i].T,
                                                G[i, i], S[i, j][l]))
                    tmp5p, tmp5n = _get_positive_negative(tmp5)
                    G_enum += tmp4p + np.dot(G_t, tmp5n)
                    G_denom += tmp4n + np.dot(G_t, tmp5p)

        for r, thetas_p in thetas_pos.items():
            if r != _type:
                continue
            for theta_p in thetas_p:
                G_denom[r, r] += np.dot(theta_p, G[r, r])

        for r, thetas_n in thetas_neg.items():
            if r != _type:
                continue
            for theta_n in thetas_n:
                G_enum += np.dot(t_n, G_t)

        G_t = np.multiply(G_t, np.sqrt(
            np.divide(G_enum, np.maximum(G_denom, np.finfo(np.float).eps))))

        s = 0
        for r in R:
            i, j = r
            for l in range(len(R[r])):
                if i is _type:
                    Rij_app = np.dot(G_t, np.dot(S[i, j][l], G[j, j].T))
                    r_err = np.linalg.norm(R[r][l]-Rij_app, "fro")
                if j is _type:
                    Rij_app = np.dot(G[i, i], np.dot(S[i, j][l], G_t.T))
                    r_err = np.linalg.norm(R[r][l]-Rij_app, "fro")
                if verbose > 1:
                    print("Relation R_%s,%s^(%d) norm difference: "
                          "%5.4f" % (i, j, l, r_err))
                s += r_err

        obj.append(s)
        err.append(s)
        if verbose:
            print("Iter: %d\t obj: %5.4f\t diff_obj: %4f" %
                  (_iter, s, err[-1] - err[-2]))
        # convergence check
        if np.abs(err[-1] - err[-2]) < tol:
            break
    else:
        warnings.warn("The algorithm did not converge", Warning)

    if verbose:
        print("Violations of optimization objective: %d/%d " % (
              int(np.sum(np.diff(obj) > 0)), len(obj)))
    result = dict(G_t=G_t, n_iter=_iter)
    return result
