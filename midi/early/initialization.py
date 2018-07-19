from operator import itemgetter
import numpy as np

from sklearn.cluster import k_means


def _initialize(dimensions, ranks, Rs, init, random_state):
    Gs = dict()
    if init == 'kmeans':
        for t in dimensions.keys():
            n, k = dimensions[t], ranks[t]
            Gs[t, t] = np.zeros((n, k))
            how_many_relation = 0
            for r in Rs:
                if r[0] == t:
                    labels = k_means(Rs[r], k)[1]
                    for i in range(n):
                        Gs[t, t][i, labels[i]] += 1
                    how_many_relation += 1
                elif r[1] == t:
                    labels = k_means(Rs[r].T, k)[1]
                    for i in range(n):
                        Gs[t, t][i, labels[i]] += 1
                    how_many_relation += 1
            Gs[t, t] /= how_many_relation   # makes the mean between the
                                            # clustering of all the relations
                                            # involving type t
        return Gs
    else:
        R_to_init = {k: V[0] for k, V in Rs.items()}
        init_types = {"random": _random, "random_c": _random_c,
                      "random_vcol": _random_vcol}
        return init_types[init](dimensions, ranks, R_to_init, random_state)


def _random(dimensions, ranks, R, random_state):
    G = {}
    for t in dimensions.keys():
        ni = dimensions[t]
        ci = ranks.get(t)
        if ci is None:
            raise ValueError("Missing type in ranks")
        G[t, t] = random_state.rand(ni, ci)
    return G


def _random_c(dimensions, ranks, R, random_state):
    G = {}
    for t in dimensions.keys():
        ni = dimensions[t]
        ci = ranks.get(t)
        if ci is None:
            raise ValueError("Missing type in ranks")
        G[t, t] = 1e-5 * np.ones((ni, ci))

        for types, Rij in R.items():
            if t not in types:
                continue
            Rij = Rij if t == types[0] else Rij.T
            p_c = int(.2 * Rij.shape[1])
            l_c = int(.5 * Rij.shape[1])
            cols_norm = [np.linalg.norm(Rij[:, i], 2)
                         for i in range(Rij.shape[1])]
            top_c = sorted(enumerate(cols_norm), key=itemgetter(1),
                           reverse=True)[:l_c]
            top_c = list(list(zip(*top_c))[0])
            Gi = np.zeros(G[t, t].shape)
            for i in range(ci):
                random_state.shuffle(top_c)
                Gi[:, i] = Rij[:, top_c[:p_c]].mean(axis=1)
            G[t, t] += np.abs(Gi)
    return G


def _random_vcol(dimensions, ranks, R, random_state):
    G = {}
    for t in dimensions.keys():
        ni = dimensions[t]
        ci = ranks.get(t)
        if ci is None:
            raise ValueError("Missing type in ranks")
        G[t, t] = 1e-5 * np.ones((ni, ci))

        for types, Rij in R.items():
            if t not in types:
                continue
            Rij = Rij if t == types[0] else Rij.T
            p_c = int(.2 * Rij.shape[1])
            Gi = np.zeros(G[t, t].shape)
            idx = np.arange(Rij.shape[1])
            for i in range(ci):
                random_state.shuffle(idx)
                Gi[:, i] = Rij[:, idx[:p_c]].mean(axis=1)
            G[t, t] += np.abs(Gi)
    return G
