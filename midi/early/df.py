import warnings
import numpy as np

from sklearn.utils import check_array, check_random_state
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import silhouette_score, accuracy_score

from .initialization import _initialize
from .utils import _check_dimensions_objects
from .optimization import _dfmf, _transform


class DFMF(BaseEstimator, ClusterMixin, TransformerMixin):
    """
    max_iter: int, optional, default:200
        Maximum number of iteration for the fit function

    init: string, optional default:'kmeans'
        Method of initialization of the Gs matrices.
        Options:
            - kmeans

    tol: float, optional default: 1e-3

    stopping: tuple, optional default=None

    n_jobs: int, optional default=1

    verbose: int, optional
        If the fit function gives output

    random_state: int or numpy.RandomState
    """

    def __init__(self, ranks, max_iter=200, init='kmeans', tol=1e-3,
                 stopping=None, n_jobs=1, verbose=0, random_state=None):
        self.ranks = ranks
        self.max_iter = max_iter
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.stopping = stopping
        self.tol = tol
        self.n_jobs = n_jobs

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        for r in X:
            for l in range(len(X[r])):
                X[r][l] = check_array(X[r][l], accept_sparse='csr',
                                      dtype=[np.float64, np.float64])
                if X[r][l].shape[0] < self.n_clusters[r[0]]:
                    raise ValueError("n_samples=%d should be >= n_clusters=%d"
                                     % (X[r][l].shape[0],
                                        self.n_clusters[r[0]]))
                if X[r][l].shape[1] < self.n_clusters[r[1]]:
                    raise ValueError("n_samples=%d should be >= n_clusters=%d"
                                     % (X[r][l].shape[1],
                                        self.n_clusters[r[1]]))
        return X

    def _check_ranks(self, types):
        for t in types:
            if self.ranks.get(t) is None:
                ValueError("Missin the rank of the type %s" % t)
        return self.ranks

    def _get_clusters(self, Gs=None):
        if Gs is None:
            Gs = self.G_
        if isinstance(Gs, dict):
            clusters = dict()
            for k in Gs.keys():
                G = Gs[k]
                clusters[k[0]] = np.argmax(G, axis=1)
            return clusters
        else:
            clusters = np.argmax(Gs, axis=1)
            return clusters

    def fit(self, Rs, Thetas=dict()):
        """

        :param Rs:
        :param ranks:
        :param Thetas:
        :return:
        """

        dimensions = _check_dimensions_objects(Rs)
        #print(dimensions['patients'])
        types = list(dimensions.keys())
        ranks = self._check_ranks(types)
        self.n_clusters = ranks
        Rs = self._check_fit_data(Rs)
        random_state = check_random_state(self.random_state)

        Gs = _initialize(dimensions, ranks, Rs, self.init, random_state)
        S = None

        _result = _dfmf(Rs, Gs, S, Thetas, types, max_iter=self.max_iter,
                        stopping=self.stopping, verbose=self.verbose,
                        tol=self.tol, n_jobs=self.n_jobs)
        self.G_ = _result['G']
        self.S_ = _result['S']
        self.n_iter_ = _result['n_iter']
        self.labels_ = self._get_clusters()
        self.ranks = ranks

        return self

    def fit_predict(self, Rs, Thetas=dict()):
        """Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        y : Ignored
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        return self.fit(Rs, Thetas).labels_

    def transform(self, Rs, Thetas, _type):
        """Transform X to a cluster-distance space.
        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        check_is_fitted(self, 'labels_')
        return self._transform(Rs, Thetas, _type)

    def _transform(self, Rs, Thetas, _type):
        """
        :param Rs:
        :param Thetas:
        :param _type:
        :return:
        """
        targets = []
        for i, j in Rs:
            entry = 0 if _type == i else 1
            targets.append(Rs[i, j][0].shape[entry])

        if len(set(targets)) > 1:
            warnings.warn("Target object type: %s size mismatch" % _type,
                          Warning)

        targets = targets[0]
        dimensions = _check_dimensions_objects(Rs)
        Gx = _initialize(dimensions, self.ranks, Rs, self.init,
                         self.random_state)
        G_t = Gx[_type, _type]

        _result = _transform(Rs, Thetas, _type, ranks=self.ranks, G=self.G_,
                             S=self.S_, G_t=G_t, max_iter=self.max_iter,
                             verbose=self.verbose, tol=self.tol,
                             random_state=self.random_state)

        return _result['G_t'], self._get_clusters(_result['G_t'])

    def predict(self, Rs, Thetas, _type):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, 'labels_')

        _, labels = _transform(self, Rs, Thetas, _type)
        return labels

    def score(self, Rs, Thetas, _type, y=None):
        """Opposite of the value of X on the K-means objective.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data.
        y : Ignored
        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self, 'labels_')

        Rs = self._check_fit_data(Rs)
        _, labels = _transform(self, Rs, Thetas, _type)

        total_relations = 0
        silhouette = 0
        for r in Rs:
            for l in range(len(Rs)):
                if r[0] == _type:
                    silhouette += silhouette_score(Rs[r][l], labels)
                    total_relations += 1
                elif r[1] == _type:
                    silhouette += silhouette_score(Rs[r][l].T, labels)
                    total_relations += 1
        silhouette /= total_relations

        if y is not None:
            accuracy = accuracy_score(y, labels)
            ret = (silhouette, accuracy)
        else:
            ret = (silhouette)

        return ret
