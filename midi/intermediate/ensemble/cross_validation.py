import warnings

from sklearn.base import BaseMixin


class EnsembleClassifierCV(BaseMixin):
    """Base RandomCV.

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

    def fit(self, X, y):
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

        check_consistent_length(X)
        check_consistent_length(X[0], y)
        self.random_state = check_random_state(self.random_state)
        D = len(X)

        if self.estimators.lower() != 'elasticnet':
            raise ValueError("Estimators different than ElasticNet are not "
                             "allowed.")
            sys.exit(0)

        if self.estimators.lower() == 'elasticnet':
            if (len(self.estimator_params) != 1
               and len(self.estimator_params) != D+1):
                raise ValueError("ElasticNet estimator wants two parameters, "
                                 "alphas and l1_ratios. Read documentation.")
            if len(self.estimator_params[0]) == 1:
                alphas = self.estimator_params[0]*(D+1)
            elif len(self.estimator_params) == D+1:
                alphas = [self.estimator_params[i][0] for i in range(D+1)]
            else:
                raise ValueError("Alphas parameter can be either one repeited "
                                 "for all or a list long as the number of "
                                 "input datasets plus one.")

            if len(self.estimator_params[1]) == 1:
                l1_ratios = self.estimator_params[1]*(D+1)
            elif len(self.estimator_params) == D+1:
                l1_ratios = [self.estimator_params[i][1] for i in range(D+1)]
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
        predictions = np.zeros_like(R[:, 0])
        for i in range(len(self.coef_)):
            predictions += self.coef_[i]*R[:, i]
        return np.sign(predictions)

    def score(self, X, y):
        predictions = self.predict(X)
        return balanced_accuracy_score(y, predictions)
