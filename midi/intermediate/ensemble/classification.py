# Authors: Veronica Tozzo (veronica.tozzo@dibris.unige.it)
#
# License: BSD 3 clause

from __future__ import division

import warnings
import numpy as np
import sys

from midi.utils import convergence, prox_elastic_net, prox_l2_el
from sklearn.base import ClassifierMixin
# from sklearn.externals.joblib import Parallel, delayed

from sklearn.utils import check_array, check_consistent_length, \
                            check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import balanced_accuracy_score

__all__ = []


def _objective_function(w, W, X, y, taus, mus):
    R = np.array([w_i.dot(X_i.T) for w_i, X_i in zip(W, X)]).T
    obj = np.linalg.norm(y - R.dot(w))**2
    obj += taus[0]*np.linalg.norm(w, 1) + mus[0]*np.linalg.norm(w)**2
    for i in range(len(W)):
        obj += np.linalg.norm(y - X[i].dot(W[i]))**2
        obj += taus[i+1]*np.linalg.norm(W[i], 1) \
            + mus[i+1]*np.linalg.norm(W[i])**2
    return obj


def _fit_ensemble_en(X, y, alphas=0.01, l1_ratios=0.5, max_iter=200,
                     random_state=None, tol=1e-3, rtol=1e-3, verbose=0):
    D = len(X)
    taus = alphas*l1_ratios
    mus = alphas*(1 - l1_ratios)

    w = random_state.randn(D)
    W = [random_state.randn(p_i)/1e5 for p_i in [X_i.shape[1] for X_i in X]]

    gamma_i = [1e-10 for X_i in X]#[2/np.linalg.norm(X_i.T.dot(X_i)) for X_i in X]
    obj = float("inf")
    checks = []
    for _iter in range(max_iter):
        w_old = w.copy()
        W_old = W.copy()
        obj_old = obj
        R = np.array([w_i.dot(X_i.T) for w_i, X_i in zip(W, X)]).T
        gamma = 1e-10#2/np.linalg.norm(R) #probably not
        #print(gamma)
        grad_w = - R.T.dot(y-R.dot(w))
        w = prox_elastic_net(w - gamma*grad_w, gamma*taus[0], gamma*mus[0])
        #print(w)
        #next done sequentially but can be done in Parallel
        Rw = R.dot(w)
        grad_W = [- w[i]*X[i].T.dot(y - Rw) for i in range(len(W))]

        for i in range(len(W)):
            W[i] = prox_l2_el(W[i] - gamma_i[i]*grad_W[i], X[i], y,
                              taus[i+1], mus[i+1], gamma_i[i], max_iters=200)
        #print(W)
        obj = _objective_function(w, W, X, y, taus, mus)
        iter_diff = (np.linalg.norm(w - w_old) +
                     np.sum([np.linalg.norm(W[i] - W_old[i])
                            for i in range(len(W))]))

        check = convergence(obj, iter_diff, [w, W])
        checks.append(check)

        if verbose:
            print("Iter: %d,\t obj:%.2f,\t obj_diff:%.5f,\t iter_diff: %.5f"%
                  (_iter, check[0], obj_old - obj, check[1]))
        if np.abs(obj_old - obj) < tol:
            break
    else:
        warnings.warn("The optimization algorithm for EEN did not converge.")

    return w, W, (checks, _iter)
# def _fit_ensemble_en(X, y, alphas=0.01, l1_ratios=0.5, max_iter=200,
#                      random_state=None, tol=1e-3, rtol=1e-3, verbose=0):
#     D = len(X)
#     taus = alphas*l1_ratios
#     mus = alphas*(1 - l1_ratios)
#
#     w = random_state.randn(D)
#     W = [random_state.randn(p_i)/1e5 for p_i in [X_i.shape[1] for X_i in X]]
#
#     gamma_i = [2/np.linalg.norm(X_i.T.dot(X_i)) for X_i in X]
#     obj = float("inf")
#     checks = []
#     for _iter in range(max_iter):
#         w_old = w.copy()
#         W_old = W.copy()
#         obj_old = obj
#         R = np.array([w_i.dot(X_i.T) for w_i, X_i in zip(W, X)]).T
#         gamma = 2/np.linalg.norm(R) #probably not
#         print(gamma)
#         grad_w = - R.T.dot(y-R.dot(w)) + mus[0]*w
#         w = soft_thresholding(w - gamma*grad_w, gamma*taus[0])
#
#         #next done sequentially but can be done in Parallel
#         Rw = R.dot(w)
#         grad_W = [- w[i]*X[i].T.dot(y - Rw)
#                   - X[i].T.dot(y - X[i].dot(W[i])) + mus[i+1]*W[i]
#                   for i in range(len(W))]
#
#         for i in range(len(W)):
#             W[i] = soft_thresholding(W[i] - gamma_i[i]*grad_W[i],
#                                      gamma_i[i]*taus[i+1])
#
#         obj = _objective_function(w, W, X, y, taus, mus)
#         iter_diff = (np.linalg.norm(w - w_old) +
#                      np.sum([np.linalg.norm(W[i] - W_old[i])
#                             for i in range(len(W))]))
#
#         check = convergence(obj, iter_diff, [w, W])
#         checks.append(check)
#
#         if verbose:
#             print("Iter: %d,\t obj:%.2f,\t obj_diff:%.2f,\t iter_diff: %.2f"%
#                   (_iter, check[0], obj_old - obj, check[1]))
#         if np.abs(obj_old - obj) < tol:
#             break
#     else:
#         warnings.warn("The optimization algorithm for EEN did not converge.")
#
#     return w, W, (checks, _iter)


class EnsembleClassifier(ClassifierMixin):
    """Base class for Ensemble classifier.


    estimator_params: a "
                     "list of alphas (regularization coefficients) "
                     "and a list of l1_ratios. The two list must be "
                     "long "
    """

    def __init__(self, estimators='ElasticNet',
                 estimator_params=tuple(), n_jobs=1, max_iter=200,
                 random_state=None, tol=1e-3, rtol=1e-3,
                 verbose=0, class_weight=None):

        self.estimators = estimators
        self.max_iter = max_iter
        self.estimator_params = estimator_params
        self.n_jbos = n_jobs
        self.random_state = random_state
        self.tol = tol
        self.rtol = rtol
        self.verbose = verbose
        self.class_weight = class_weight

    # def _validate_y_class_weight(self, y):
    #     check_classification_targets(y)
    #
    #     y = np.copy(y)
    #
    #     for k in range(self.n_outputs_):
    #         classes_k, _ = np.unique(y[:, k], return_inverse=True)
    #         self.n_classes_.append(classes_k.shape[0])
    #
    #     if self.n_classes != 2:
    #         raise ValueError("The estimator works only on 2-classes scenarios")
    #     return y

    def fit(self, X, y, max_iter=500):
        """
        X: list of array, length D, shape of each array i
            (n_samples, n_features_i).
            The list of different datasets to use for the prediction of the
            output y.
        y: array-like, shape=(n_samples, )

        """
        if type(X) != list:
            raise ValueError("X must be a list of datasets, found the type "
                             + str(type(X)))
        X = [check_array(X_i) for X_i in X]
        self.dimensions = [X_i.shape[1] for X_i in X]

        #y = check_array(y)# self._validate_y_class_weight(y)
        check_consistent_length(X)
        check_consistent_length(X[0], y)
        self.random_state = check_random_state(self.random_state)
        D = len(X)

        if self.estimators.lower() != 'elasticnet':
            raise ValueError("Estimators different than ElasticNet are not "
                             "allowed.")
            sys.exit(0)

        if self.estimators.lower() == 'elasticnet':
            if len(self.estimator_params) != 2:
                raise ValueError("ElasticNet estimator wants two parameters, "
                                 "alphas and l1_ratios")
            if len(self.estimator_params[0]) == 1:
                alphas = self.estimator_params[0]*(D+1)
            elif len(self.estimator_params) == D+1:
                alphas = self.estimator_params[0]
            else:
                raise ValueError("Alphas parameter can be either one repeited "
                                 "for all or a list long as the number of "
                                 "input datasets plus one.")

            if len(self.estimator_params[1]) == 1:
                l1_ratios = self.estimator_params[1]*(D+1)
            elif len(self.estimator_params) == D+1:
                l1_ratios = self.estimator_params[1]
            else:
                raise ValueError("L1_ratios parameter can be either one "
                                 "repeited for all or a list long as the "
                                 "number of input datasets plus one.")

            alphas = np.array(alphas)
            l1_ratios = np.array(l1_ratios)
            self.coef_, self.W_, self.history = _fit_ensemble_en(
                X, y,  alphas=alphas, l1_ratios=l1_ratios,
                max_iter=self.max_iter, random_state=self.random_state,
                tol=self.tol, rtol=self.rtol, verbose=self.verbose)

        return self

    def _check_consistency_with_train(self, X):
        if len(X) != len(self.dimensions):
            raise ValueError("The number of input datasets must be %d" %
                             len(self.dimensions))
        for i in range(len(X)):
            if X[i].shape[1] != self.dimensions[i]:
                raise ValueError("The number of features of the datasets must "
                                 "be consistent with the training.")

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is the sign of the inner
        representation combined with the ensemble representation.

        Parameters
        ----------
        X : list of array-like, length D, shape_i = [n_samples, n_features_i]

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """
        # controlla che sia fittato
        if type(X) != list:
            raise ValueError("X must be a list of datasets, found the type "
                             + str(type(X)))
        X = [check_array(X_i) for X_i in X]
        check_is_fitted(self, ["coef_"])
        self._check_consistency_with_train(X)
        R = np.array([w_i.dot(X_i.T) for w_i, X_i in zip(self.W_, X)]).T
        predictions = np.sign(R.dot(self.coef_))
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        return balanced_accuracy_score(y, predictions)
