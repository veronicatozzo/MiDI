from operator import add
from functools import reduce

import numpy as np
from joblib import Parallel, delayed

from numbers import Number
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import ParameterGrid


def _shuffle_split(estimator, R, y, _t,  Thetas=dict(), n_splits=5,
                   test_size=0.3):
    #t1, t2 = estimator.predictive_relationship
    dims = _check_dimensions_objects2(R)
    n = dims[_t]

    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size)

    scores = []
    for train, test in ss.split(np.arange(0, n)):
        Rtr = dict()
        Rts = dict()
        for r in R:
            if r[0] == _t:
                Rtr[r] = R[r][train, :]
                Rts[r] = R[r][test, :]
            elif r[1] == _t:
                Rtr[r] = R[r][:, train]
                Rts[r] = R[r][:, test]
            else:
                Rtr[r] = R[r]
                Rts[r] = R[r]

        Ttr = dict()
        Tts = dict()
        for t in Thetas:
                if t != _t:
                    Ttr[t] = Thetas[t]
                    Tts[t] = Thetas[t]
                else:
                    Ttr[t] = [Thetas[t][l][train, :]
                              for l in range(len(Thetas[t]))]
                    Tts[t] = [Thetas[t][l][test, :]
                              for l in range(len(Thetas[t]))]
        if y is not None:
            ytr = y[train, :]
            yts = y[test, :]
            estimator.fit(Rtr, Ttr)
            scores.append(estimator.score(Rts, _t, Tts))
        else:
            estimator.fit(Rtr, Ttr)
            scores.append(estimator.score(Rts, _t, Tts))

    return np.mean(scores)


def grid_search(estimator, R, grid, y, t,  Thetas=dict(), n_splits=5,
                test_size=0.3, verbose=0):
        """the grid is a dictionary where for each type its specified a list
        of possible ranks to consider"""
        possible_ranks = list(ParameterGrid(grid))
        best_ranks = None
        best_score = - np.inf
        best_estimator = None
        results = []
        for i, ranks in enumerate(possible_ranks):
            estimator.set_params(ranks=ranks)
            score = _shuffle_split(estimator, R, y, t, Thetas=Thetas,
                                   n_splits=n_splits, test_size=test_size)
            if verbose:
                print(ranks)
                print("Done iteration %d with schore %4f" % (i, score))
            results.append([estimator, score])
            if score > best_score:
                best_score = score
                best_ranks = ranks
                best_estimator = estimator
        return best_estimator, best_ranks, best_score, results


def _relations_norm(R):
    norm = 0
    for r in R:
        for l in range(len(R[r])):
            norm += np.linalg.norm(R[r][l])
    return norm


def fill_mean(x):
    mean = np.nanmean(x)
    if np.ma.is_masked(x):
        indices = np.logical_or(~np.isfinite(x), x.mask)
    else:
        indices = ~np.isfinite(x)
    filled = x.copy()
    filled[indices] = mean
    return filled


def fill_row(x):
    row_mean = np.nanmean(x, 1)
    mat_mean = np.nanmean(x)
    if np.ma.is_masked(x):
        # default fill_value in Numpy MaskedArray is 1e20.
        # mean gets masked if entire rows are unknown
        row_mean = np.ma.masked_invalid(row_mean)
        row_mean = np.ma.filled(row_mean, mat_mean)
        indices = np.logical_or(~np.isfinite(x.data), x.mask)
    else:
        row_mean[np.isnan(row_mean)] = mat_mean
        indices = ~np.isfinite(x)
    filled = x.copy()
    filled[indices] = np.take(row_mean, indices.nonzero()[0])
    return filled


def fill_col(x):
    return fill_row(x.T).T


def fill_const(x, const):
    filled = x.copy()
    filled[~np.isfinite(x)] = const
    if np.ma.is_masked(x):
        filled.data[x.mask] = const
    return filled


def filled(R, fill_value='mean'):
    FILL_CONST = 'const'
    FILL_TYPE = dict([
        ('mean', fill_mean),
        ('row_mean', fill_row),
        ('col_mean', fill_col),
        ('const', fill_const)
    ])
    if isinstance(fill_value, Number):
        filled_data = FILL_TYPE[FILL_CONST](R, fill_value)
    else:
        filled_data = FILL_TYPE[fill_value](R)
    return filled_data


def __bdot(A, B, i, j, obj_types):
    entry = []
    if isinstance(list(A.values())[0], list):
        for l in range(len(A.get((i, j), []))):
            ll = [np.dot(A[i, k][l], B[k, j]) for k in obj_types
                  if (i, k) in A and (k, j) in B]
            if len(ll) > 0:
                tmp = reduce(add, ll)
                entry.append(np.nan_to_num(tmp))
    elif isinstance(list(B.values())[0], list):
        for l in range(len(B.get((i, j), []))):
            ll = [np.dot(A[i, k], B[k, j][l]) for k in obj_types
                  if (i, k) in A and (k, j) in B]
            if len(ll) > 0:
                tmp = reduce(add, ll)
                entry.append(np.nan_to_num(tmp))
    else:
        ll = [np.dot(A[i, k], B[k, j]) for k in obj_types
              if (i, k) in A and (k, j) in B]
        if len(ll) > 0:
            entry = reduce(add, ll)
            entry = np.nan_to_num(entry)
    return i, j, entry


def _par_bdot(A, B, obj_types, verbose, n_jobs):
    """Parallel block matrix multiplication.

    Parameters
    ----------
    A : dictionary of array-like objects
        Block matrix.

    B : dictionary of array-like objects
        Block matrix.

    obj_types : array-like
        Identifiers of object types.

    verbose : int
         The amount of verbosity.

    n_jobs: int (default=1)
        Number of jobs to run in parallel

    Returns
    -------
    C : dictionary of array-like objects
        Matrix product, A*B.
    """
    parallelizer = Parallel(n_jobs=n_jobs, max_nbytes=1e3, verbose=verbose,
                            backend="multiprocessing")
    task_iter = (delayed(__bdot)(A, B, i, j, obj_types)
                 for i in obj_types for j in obj_types)
    entries = parallelizer(task_iter)
    C = {(i, j): entry for i, j, entry in entries if entry != []}
    return C


def _transpose(A):
    """Block matrix transpose.

    Parameters
    ----------
    A : dictionary of array-like objects
        Block matrix.

    Returns
    -------
    At : dictionary of array-like objects
        Block matrix with each of its block transposed.
    """
    At = {k: V.T for k, V in A.items()}
    return At


def _check_dimensions_objects2(R):
    """
    :param R: dictionary from (type, type) to  relational matrix
    :return: dictionary string to int (dimension of that type)
    """

    dimensions = {}
    for r in R:
        t1, t2 = r
        if dimensions.get(t1) is not None:
            if R[r].shape[0] != dimensions[t1]:
                raise ValueError("Object type " + t1 + " has"
                                 "mismatching dimensions")
        else:
            dimensions[t1] = R[r].shape[0]

        if dimensions.get(t2) is not None:
            if R[r].shape[1] != dimensions[t2]:
                raise ValueError("Object type " + t2 + " has"
                                 "mismatching dimensions")
        else:
            dimensions[t2] = R[r].shape[1]
    return dimensions


def _check_dimensions_objects(R):
    """
    :param R: dictionary from (type, type) to  relational matrix
    :return: dictionary string to int (dimension of that type)
    """
    print(R)
    dimensions = {}
    for r in R:
        t1, t2 = r
        for l in range(len(R[r])):
            if dimensions.get(t1) is not None:
                if R[r][l].shape[0] != dimensions[t1]:
                    raise ValueError("Object type " + t1 + " has"
                                     "mismatching dimensions")
            else:
                dimensions[t1] = R[r][l].shape[0]

            if dimensions.get(t2) is not None:
                if R[r][l].shape[1] != dimensions[t2]:
                    raise ValueError("Object type " + t2 + " has"
                                     "mismatching dimensions")
            else:
                print(R[r][l].shape)
                dimensions[t2] = R[r][l].shape[1]
    return dimensions


def _get_positive_negative(X):
    pos = X > 0
    X_pos = np.multiply(pos, X)
    X_neg = np.multiply(pos-1, X)
    return X_pos, X_neg


def _compute_error(R, G, S, verbose):
    s = 0
    for r in R:
        i, j = r
        for l in range(len(R[r])):
            Rij_reconstr = np.linalg.multi_dot((G[i, i], S[i, j][l],
                                                G[j, j].T))
            r_err = np.linalg.norm(R[r][l] - Rij_reconstr, "fro")
            if verbose > 1:
                print("R_%s,%s^(%d) norm difference: %5.4f" %
                      (i, j, l, r_err))
            s += r_err
    return s
