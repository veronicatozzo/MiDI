# Authors: Veronica Tozzo (veronica.tozzo@dibris.unige.it)
#
# License: BSD 3 clause

from __future__ import division

import warnings
import numpy as np

from ..base import ClassifierMixin
from ..externals.joblib import Parallel, delayed

from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

__all__ = []


def _fit_ensemble_rls(X, y, alphas, l1_ratios, )

class EnsembleClassifier(ClassifierMixin):
    """Base class for Ensemble classifier.
    """

    def __init__(self, estimators='RLS',
                 estimator_params=tuple(), n_jobs=1, random_state=None,
                 verbose=0, class_weight=None):

        self.estimators = estimators
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params
        self.n_jbos = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight

    def _validate_y_class_weight(self, y):
        check_classification_targets(y)

        y = np.copy(y)

        for k in range(self.n_outputs_):
            classes_k, _ = np.unique(y[:, k], return_inverse=True)
            self.n_classes_.append(classes_k.shape[0])

        if self.n_classes != 2:
            raise ValueError("The estimator works only on 2-classes scenarios")
        return y

    def fit(self, X, y):
        """
        X: list of array, length D, shape of each array i
            (n_samples, n_features_i).
            The list of different datasets to use for the prediction of the
            output y.
        y: array-like, shape=(n_samples, )
        
        """
        X = check_array(X)
        y = self._validate_y_class_weight(y)
        check_consistent_length(X, y)

        if estimators.lower() != 'elasticnet':
            warnings.warn("Estimators different than ElasticNet are not "
                          "allowed.")
            sys.exit(0)



    def predict(self, X):
        """Predict class for X.
        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            predictions = np.zeros((n_samples, self.n_outputs_))

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(np.argmax(proba[k],
                                                                    axis=1),
                                                          axis=0)

            return predictions

    def predict_proba(self, X):
        """Predict class probabilities for X.
        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest. The
        class probability of a single tree is the fraction of samples of the same
        class in a leaf.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        check_is_fitted(self, 'estimators_')
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [np.zeros((X.shape[0], j), dtype=np.float64)
                     for j in np.atleast_1d(self.n_classes_)]
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, backend="threading")(
            delayed(accumulate_prediction)(e.predict_proba, X, all_proba, lock)
            for e in self.estimators_)

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.
        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the trees in the
        forest.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba
