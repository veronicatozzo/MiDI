import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from lifelines import AalenAdditiveFitter, KaplanMeierFitter

from sklearn.utils import check_array, check_random_state
from sklearn.base import BaseEstimator

from .df import DFMF
from .utils import _compute_error, _relations_norm


class DFSP(DFMF, AalenAdditiveFitter, KaplanMeierFitter, BaseEstimator):

    def __init__(self, predictive_relationship, ranks, max_iter=200,
                 init='kmeans', tol=1e-3, stopping=None, n_jobs=1, verbose=0,
                 random_state=None, coef_penalizer=1.0, fit_intercept=True):
        self.predictive_relationship = predictive_relationship
        AalenAdditiveFitter.__init__(
            self, coef_penalizer=coef_penalizer, fit_intercept=fit_intercept)
        DFMF.__init__(
            self, ranks=ranks, max_iter=max_iter, init=init, tol=tol,
            stopping=stopping,  n_jobs=n_jobs, verbose=verbose,
            random_state=random_state)

    def _check_y(self, y, G_t=None):
        t, _ = self.predictive_relationship
        if G_t is None:
            G_t = self.G_[t, t]
        try:
            n, c = y.shape
        except ValueError:
            y = y[:, np.newaxis]
            n, c = y.shape
        y = check_array(y)
        if n != G_t.shape[0]:
            raise ValueError("The first dimension of the predictive "
                             "relationship %d and the output %d must"
                             "be the same" % (G_t.shape[0], n))
        if c > 2:
            raise ValueError("Not expecting an y with more than two "
                             "columns. Found %d" % c)
        return y

    def _fit_Kaplan_Meier(self, partition=None, labels=None):
        if self.verbose:
            print("Fitting Kaplan Meier model")
        if partition is None:
            t, _ = self.predictive_relationship
            partition = self.labels_[t]

        if labels is not None:
            print(len(labels))
            print(len(np.unique(partition)))
            if len(labels) != len(np.unique(partition)):
                raise ValueError("The labels have to be the same length as"
                                 "the number of groups")
        else:
            labels = {}
            for c in np.unique(partition):
                labels[c] = "Group "+str(c)

        groups = dict()
        for c in np.unique(partition):
            indices = np.where(partition == c)
            if self.outputs_.size > self.outputs_.shape[0]:
                kmf = KaplanMeierFitter()
                duration = self.outputs_[indices, 0].reshape(
                                                        indices[0].shape[0])
                kmf.fit(duration,
                        event_observed=list(self.outputs_[indices, 1]),
                        label=labels[c])
                groups[c] = kmf
            else:
                kmf = KaplanMeierFitter()
                duration = self.outputs_[indices, 0].reshape(
                                                        indices[0].shape[0])
                kmf.fit(duration)
                groups[c] = kmf
        self.kaplan_meier_ = groups

    def fit(self, R, y=None, Thetas=dict()):
        """
        R: dict
            Relations between types

        ranks: dict
            Number of latent factors in which to decompose the relations

        y: array-like, dimensions (n x 1) or (n x 2)
            The first column is the time to predict.
            The second column is optional and is the event we are predicting.
        """
        self.random_state = check_random_state(self.random_state)
        if self.verbose:
            print("Fitting data fusion procedure")
        DFMF.fit(self, R, Thetas)
        if y is not None:
            y = self._check_y(y)
            self.outputs_ = y

        t1, t2 = self.predictive_relationship
        factors = ["Factor " + str(i) for i in range(1, self.ranks[t2] + 1)]
        x = self.G_[t1, t1].dot(self.S_[t1, t2][0])
        X = pd.DataFrame(x, columns=factors)

        if y is not None:
            self.regression_ = True
            if self.verbose:
                print("Fitting Aalen additive model")
            if y.shape[1] == 2:
                X['T'] = y[:, 0]
                X['E'] = y[:, 1]
                AalenAdditiveFitter.fit(self, X, 'T', event_col='E')
            else:
                X['T'] = y
                AalenAdditiveFitter.fit(self, X, 'T')
            self._fit_Kaplan_Meier()
        else:
            self.regression_ = False

        return self

    def _modify_test_data(self,  R, Thetas):
        res = DFMF.transform(self, R, Thetas=Thetas,
                             _type=self.predictive_relationship[0])
        t1, t2 = self.predictive_relationship
        factors = ["Factor " + str(i) for i in range(1, self.n_clusters[t2]+1)]
        x = res[0].dot(self.S_[t1, t2][0])
        X = pd.DataFrame(x, columns=factors)
        return X

    def predict(self, R, Thetas=dict(), _type='cumulative_hazards', **kwargs):
        """
        Assuming that the type to refit is the first type of
         predictive_relationship
        """

        if not self.regression_:
            raise Exception("No regression was fitted on the traning")

        X = self._modify_test_data(R, Thetas)
        if _type == 'cumulative_hazards':
            return AalenAdditiveFitter.predict_cumulative_hazard(
                    self, X, id_col=kwargs.get('id_col', None))
        elif _type == 'survival_function':
            return AalenAdditiveFitter.predict_survival_function(self, X)
        elif _type == 'percentile':
            return AalenAdditiveFitter.predict_percentile(
                        self, X, kwargs.get('p', 0))
        elif _type == 'median':
            return AalenAdditiveFitter.predict_median(self, X)
        elif _type == 'expectation':
            return AalenAdditiveFitter.predict_expectation(self, X)
        else:
            raise ValueError("Not avaialble type of prediction")

    def plot_Kaplan_Meier(self, figsize=(10, 15), ylim=(0, 1), xlim=None,
                          title="Kaplan-Meier", **kwargs):
        """

        Other args are:
        show_censors: place markers at censorship events. Default: False
        censor_styles: If show_censors, this dictionary will be passed into
                       the plot call.
        ci_alpha: the transparency level of the confidence interval.
                  Default: 0.3
        ci_force_lines: force the confidence intervals to be line plots
                        (versus default shaded areas). Default: False
        ci_show: show confidence intervals. Default: True
        ci_legend: if ci_force_lines is True, this is a boolean flag to add
                   the lines' labels to the legend. Default: False
        at_risk_counts: show group sizes at time points. See function
                        'add_at_risk_counts' for details. Default: False
        loc: specify a time-based subsection of the curves to plot, ex:
                 .plot(loc=slice(0.,10.))
            will plot the time values between t=0. and t=10.
        iloc: specify a location-based subsection of the curves to plot, ex:
                 .plot(iloc=slice(0,10))
              will plot the first 10 time points.
        bandwidth: specify the bandwidth of the kernel smoother for the
                   smoothed-hazard rate. Only used when called 'plot_hazard'.
        """
        fig, ax = plt.subplots(1, figsize=figsize)
        for g in self.kaplan_meier_:
            self.kaplan_meier_[g].plot(ax=ax, **kwargs)
        if xlim is not None:
            plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.title(title)

    def score(self, R, Thetas, y=None):
        if y is not None:
            y_pred = self.predict(R, Thetas, 'expectation')
            res = DFMF.transform(self, R, Thetas=Thetas,
                                 _type=self.predictive_relationship[0])
            y = self._check_y(y, res[0])
            return -(np.linalg.norm(y_pred - y[:, 0][:, np.newaxis])
                     / np.linalg.norm(y))
        else:
            t, _ = self.predictive_relationship
            res = DFMF.transform(self, R, Thetas=Thetas,
                                 _type=t)
            G = self.G_
            G[t, t] = res[0]
            err = (_compute_error(R, G, self.S_, self.verbose) /
                   _relations_norm(R))
            return - (2.5*err +
                      np.sum(np.log([self.ranks[v] for v in self.ranks])))
