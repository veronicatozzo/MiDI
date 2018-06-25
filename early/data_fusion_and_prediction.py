from collections import defaultdict

import numpy as np

from sklearn.utils import check_random_state

from .initialization import initialize
from .utils import _check_dimensions_objects, _get_positive_negative, \
                  _compute_error, _update_G, _update_S, _update_Spr, \
                  _update_B


def dfmf_sr(R, ranks, I, Rpr, baseline_hazard, thetas=dict(), max_iter=100,
            init_type="random_c", stopping_relation=None, verbose=0, tol=1e-3,
            callback=None, random_state=None, n_jobs=1):
    """Data fusion by matrix factorization.

    Parameters
    ----------
    R : dictionary of array-like objects
        Relation matrices.

    ranks: dictionary of int
        The ranks of each data type.

    I: array-like
      Matrix of dimension (n_samples_type_p, n_times) where at the vector
      corresponding to a time k there are ones for all those samples who
      experience an event in that point in time.

    Rpr: tuple
        Relation to use for the prediction of the outcome I.

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
    """
    # TODO check dimension of baseline_hazard

    dimensions = _check_dimensions_objects(R)
    obj_types = list(dimensions.keys())
    random_state = check_random_state(random_state)
    # if not isinstance(random_state, np.random.RandomState):
    #     random_state = np.random.RandomState(random_state)
    R_to_init = {k: V[0] for k, V in R.items()}
    G = initialize(dimensions, ranks, R_to_init, init_type, random_state)
    S = None
    B = np.zeros((I.shape[1], ranks[Rpr[1]]))
    err = [1e10]
    p, r = Rpr

    # getting positive and negative thetas
    thetas_pos, thetas_neg = defaultdict(list), defaultdict(list)
    for t, thetas_of_t in thetas.items():
        for theta in thetas_of_t:
            pos, neg = _get_positive_negative(theta)
            thetas_pos[t].append(pos)
            thetas_neg[t].append(neg)  # meno uno non fa casini coi segni?

    obj = [1e10]

    for _iter in range(max_iter):

        S = _update_S(G, R, obj_types, n_jobs, verbose=0)
        p, r = Rpr
        S[p, r] = [_update_Spr(G[p, p], G[r, r], R[p, r],
                    S[p, r][0].shape, I, B, (_iter % I.shape[1]))]*len(R[p, r])
        G = _update_G(G, R, S, thetas_pos, thetas_neg, I, B, Rpr, verbose)
        B = _update_B(G[p, p], S[p, r], I)

        # Reconstruction error
        error = _compute_error(R, G, S, verbose)
        obj.append(error)
        if stopping_relation:
            t1, t2 = stopping_relation
            R_reconstructed = np.linalg.multi_dot((G[t1, t1],
                                                   S[t1, t2], G[t2, t2].T))
            diff = R[t] - R_reconstructed
            error = np.linalg.norm(diff)
        err.append(error)

        if callback:
            callback(G, S, iter)

        if verbose:
            print("At iteration %d, the error is (objective function value):"
                  " %5.4f" % (_iter, error))
        # convergence check
        if np.abs(obj[-1] - obj[-2]) < tol:
            if verbose:
                print("Early stopping: target matrix change < %5.4f" %
                      tol)
            break

    else:
        print("Algorithm did not converge")

    print("Violations of optimization objective: %d/%d, iter=%d" % (
                  int(np.sum(np.diff(obj) > 0)), len(obj), _iter))

    return_ = dict(G=G, S=S, n_iter=_iter, B=B)
    return return_
