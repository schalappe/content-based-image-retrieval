# -*- coding: utf-8 -*-
"""
Métriques d'évaluation d'une recherche tronquée à `k` résultats.

`k` is the length of each result list, i.e. the depth used at search time; an empty list is a failed query and
scores zero. Aggregates raise `ValueError` on mismatched lengths and `statistics.StatisticsError` on empty input.
"""

from statistics import fmean
from typing import Sequence


def reciprocal_rank(found: Sequence[str], ground_truth: str) -> float:
    """
    Inverse du rang du premier élément pertinent (RR@k), zéro si aucun.

    Parameters
    ----------
    found : Sequence[str]
        Labels trouvés, du plus proche au plus lointain.
    ground_truth : str
        Vrai label.

    Returns
    -------
    float
        Rang réciproque.
    """
    for rank, item in enumerate(found, start=1):
        if item == ground_truth:
            return 1 / rank
    return 0.0


def average_precision(found: Sequence[str], ground_truth: str, relevant: int) -> float:
    """
    Précision moyenne AP@k, normalisée par `min(relevant, k)`.

    Normalising by the number of relevant items the base could have returned (not by the hits) makes a list with
    one hit score lower than a list with five.

    Parameters
    ----------
    found : Sequence[str]
        Labels trouvés, du plus proche au plus lointain.
    ground_truth : str
        Vrai label.
    relevant : int
        Nombre d'éléments pertinents dans la base.

    Returns
    -------
    float
        Précision moyenne.
    """
    hits, total = 0, 0.0
    for rank, item in enumerate(found, start=1):
        if item == ground_truth:
            hits += 1
            total += hits / rank
    denominator = min(relevant, len(found))
    return total / denominator if denominator else 0.0


def mean_reciprocal_rank(retrievals: Sequence[Sequence[str]], labels: Sequence[str]) -> float:
    """
    Moyenne des rangs réciproques (MRR@k).

    Parameters
    ----------
    retrievals : Sequence[Sequence[str]]
        Labels trouvés pour chaque requête.
    labels : Sequence[str]
        Vrai label de chaque requête.

    Returns
    -------
    float
        MRR@k.
    """
    return fmean(reciprocal_rank(found, label) for found, label in zip(retrievals, labels, strict=True))


def first_rank_accuracy(retrievals: Sequence[Sequence[str]], labels: Sequence[str]) -> float:
    """
    Part des requêtes dont le premier résultat est pertinent.

    Parameters
    ----------
    retrievals : Sequence[Sequence[str]]
        Labels trouvés pour chaque requête.
    labels : Sequence[str]
        Vrai label de chaque requête.

    Returns
    -------
    float
        Précision au premier rang.
    """
    return fmean(bool(found) and found[0] == label for found, label in zip(retrievals, labels, strict=True))


def mean_average_precision(
    retrievals: Sequence[Sequence[str]], labels: Sequence[str], relevants: Sequence[int]
) -> float:
    """
    Moyenne des précisions moyennes (MAP@k).

    Parameters
    ----------
    retrievals : Sequence[Sequence[str]]
        Labels trouvés pour chaque requête.
    labels : Sequence[str]
        Vrai label de chaque requête.
    relevants : Sequence[int]
        Nombre d'éléments pertinents dans la base pour chaque requête.

    Returns
    -------
    float
        MAP@k.
    """
    return fmean(
        average_precision(found, label, relevant)
        for found, label, relevant in zip(retrievals, labels, relevants, strict=True)
    )
