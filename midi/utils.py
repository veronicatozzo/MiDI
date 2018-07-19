from __future__ import division

import warnings
import sys

import numpy as np
import pandas as pd

from sklearn.utils import check_array, check_consistent_length
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score

param_name = {'ridge': 'alpha',
              'svm': 'C',
              'logisticregression': 'C',
              'randomforest': 'n_estimators'}


def _get_parameters(model, params=np.logspace(-10, 10, 50)):
    model = str(model).lower()
    par = param_name.get(model)
    if par is None:
        warnings.warn("The method "+model+" is not allowed. Please choose "
                      "between Ridge, SVM, LogisticRegression, RandomForest")
        return None
    if model == 'randomforest' and params.dtype is not 'int64':
        warnings.warn("The parameters of a RandomForestClassifier must be "
                      "integers representing the number of estimators used."
                      "Casting the vector to int.")
        params = params.astype(int)
        params = params[np.where(params != 0)]
    return {par: params}


def _get_model(model):
    matching = {'ridge': RidgeClassifier(),
                'svm': LinearSVC(),
                'logisticregression': LogisticRegression(),
                'randomforest': RandomForestClassifier()}
    model = str(model).lower()
    obj = matching.get(model)
    if obj is None:
        warnings.warn("The method "+model+" is not allowed. Please choose "
                      "between Ridge, SVM, LogisticRegression, RandomForest")
        return None
    return obj


def benchmark_with_multiple_models(
        X, y,
        models=['Ridge', 'SVM', 'Logisticregression', 'RandomForest'],
        params_range=np.logspace(-10, 10, 50),
        n_of_repetitions=100, n_split_for_cv=3,
        verbose=0,
        filename='results.pkl'):
    """
    Function that given a dataset tests it with multiple models to check its
    performance in classifying y.

    Params
    ------
    X: array-like, shape(n_samples, n_features)
        The input matrix.

    y: array-like, shape(n_samples,)
        The output matrix.

    models: List of models the user wants to test.
        Available options are: Ridge, SVM, LogisticRegression, RandomForest
        Default: all

    params_range: List of parameters range to use for the models.
        Default: np.logspace(-10, 10, 50), it will be used indicriminatively
        for all the models.

    n_of_repetitions: int, default: 100
        Number of times in which to split the input dataset to test the models.

    n_split_for_cv: int, default: 3
        Number of K fold in which to split the training to tune the parameters.

    verbose: boolean, default:0
        If it set to 1 the procedure will print outputs during the execution.

    filename: string, default: results.csv
        The file in which to save the final results.
    """

    X = check_array(X)
    check_consistent_length(X, y)

    if len(params_range) == 1:
        params_range = [params_range]*len(models)

    if len(params_range) != len(models):
        warnings.warn("The length of the parameters range list must be equal "
                      "to the number of models. Found %d parameters "
                      "specification and"
                      " %d models" % (len(params_range), len(models)))
        sys.exit(0)

    columns = ['mean_score_tr', 'std_score_tr', 'mean_score_ts',
               'std_score_ts', 'parameters', 'scores_tr', 'scores_ts',
               'scores_CV', 'estimators', 'train_idx', 'test_idx']

    df_results = pd.DataFrame(columns=columns)

    for m, p in zip(models, params_range):
        model = _get_model(m)
        params = _get_parameters(m)

        if model is None or params is None:
            continue

        tr_scores, estimators, trains_idx, test_idx = [], [], [], []
        ts_scores, parameters, val_scores = [], [], []

        sss = StratifiedShuffleSplit(n_splits=n_of_repetitions)

        for train, test in sss.split(X, y):
            x_tr = X[train, :]
            y_tr = y[train]
            x_ts = X[test, :]
            y_ts = y[test]
            gscv = GridSearchCV(model, params, cv=n_split_for_cv)
            gscv.fit(x_tr, y_tr)
            model = gscv.best_estimator_
            parameters.append(gscv.best_params_[param_name[m.lower()]])
            val_scores.append(gscv.cv_results_)
            tr_scores.append(balanced_accuracy_score(y_tr,
                                                     model.predict(x_tr)))
            ts_scores.append(balanced_accuracy_score(y_ts,
                                                     model.predict(x_ts)))
            trains_idx.append(train)
            test_idx.append(test)
            estimators.append(model)

        means_tr = np.mean(tr_scores)
        std_tr = np.std(tr_scores)
        means_ts = np.mean(ts_scores)
        std_ts = np.std(ts_scores)

        if verbose:
            print("Finished with model %s with scores in train %f+/-%f and "
                  "in test %f+/-%f." % (m, means_tr, std_tr, means_ts, std_ts))
        res = pd.DataFrame([[means_tr, std_tr, means_ts, std_ts, parameters,
                             tr_scores, ts_scores, val_scores, estimators,
                             trains_idx, test_idx]],
                           columns=columns, index=[m])
        df_results = df_results.append(res)

    df_results.to_pickle(filename)
    print(df_results[:, 0:5])
