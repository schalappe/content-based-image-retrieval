# -*- coding: utf-8 -*-
"""
Recherche exacte des images les plus proches.
"""

from time import perf_counter

import numpy as np
from numpy import ndarray
from sklearn.metrics.pairwise import pairwise_distances

# ##: Names understood by `pairwise_distances`; all are distances, so smaller means closer.
MEASURES = ("cosine", "euclidean", "manhattan")


def search(gallery: ndarray, queries: ndarray, measure: str, depth: int, chunk: int = 512):
    """
    Recherche exhaustive des `depth` plus proches voisins de chaque requête.

    Brute force on purpose: results are exact, so differences between methods come from the features and the
    measure, not from an approximate index.

    Parameters
    ----------
    gallery : ndarray
        Vecteurs de la base, de forme (n_gallery, dim).
    queries : ndarray
        Vecteurs des requêtes, de forme (n_queries, dim).
    measure : str
        Une des mesures de `MEASURES`.
    depth : int
        Nombre de voisins à retourner.
    chunk : int, default: 512
        Nombre de requêtes traitées ensemble, pour borner la mémoire de la matrice des distances.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        Indices dans `gallery` (n_queries, depth), distances associées (n_queries, depth), et durée de recherche par
        requête en secondes (n_queries,), amortie sur chaque bloc.
    """
    indices, distances, seconds = [], [], []
    for start in range(0, len(queries), chunk):
        block = queries[start : start + chunk]
        begin = perf_counter()
        distance = pairwise_distances(block, gallery, metric=measure)
        nearest = np.argsort(distance, axis=1, kind="stable")[:, :depth]
        seconds.append(np.full(len(block), (perf_counter() - begin) / len(block)))
        indices.append(nearest)
        distances.append(np.take_along_axis(distance, nearest, axis=1))
    return np.concatenate(indices), np.concatenate(distances), np.concatenate(seconds)
