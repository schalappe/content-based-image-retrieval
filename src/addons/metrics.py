# -*- coding: utf-8 -*-
"""
Ensemble des métriques pour l'évaluation.
"""
from statistics import fmean
from typing import Sequence


def reciprocal_rank(found: Sequence[str], ground_truth: str) -> float:
    """
    Calcul du rang de réciprocité.

    Parameters
    ----------
    found : Sequence[str]
        Liste des éléments trouvés.
    ground_truth : str
        Vrai label.

    Returns
    -------
    float
        Rang de réciprocité.
    """
    try:
        rank = found.index(ground_truth) + 1
        rank = 1 / rank
    except ValueError:
        rank = 0.0
    return rank


def mean_reciprocal_rank(retrievals: Sequence[Sequence[str]], labels: Sequence[str]) -> float:
    """
    Calcul de la moyenne des rangs de réciprocité.

    Parameters
    ----------
    retrievals : Sequence[Sequence[str]]
        Liste des éléments trouvés.
    labels : Sequence[str]
        Listes des vrais labels.

    Returns
    -------
    float
        Moyenne des rangs de réciprocité.
    """
    reciprocal_rangs = list(
        map(lambda groups: reciprocal_rank(found=groups[0], ground_truth=groups[1]), zip(retrievals, labels))
    )
    return fmean(reciprocal_rangs)


def first_rank_accuracy(retrievals: Sequence[Sequence[str]], labels: Sequence[str]) -> float:
    """
    Calcul du pourcentage des labels corrects trouvés en premières positions.

    Parameters
    ----------
    retrievals : Sequence[Sequence[str]]
        Liste des éléments trouvés.
    labels : Sequence[str]
        Listes des vrais labels.

    Returns
    -------
    float
        Pourcentage des labels corrects trouvés en premières positions.
    """
    first_rank = list(map(lambda couple: couple[0][0] == couple[1] if couple[0] else False, zip(retrievals, labels)))
    return fmean(first_rank)


def precision(found: Sequence[str], ground_truth: str) -> float:
    """
    Calcul de la précision des labels trouvés.

    Parameters
    ----------
    found : Sequence[str]
        Liste des éléments trouvés.
    ground_truth : str
        Vrai label.

    Returns
    -------
    float
        Précision.
    """
    evaluation = list(map(lambda item: item == ground_truth, found))
    return fmean(evaluation)


def average_precision(found: Sequence[str], ground_truth: str) -> float:
    """
    Calcul de la précision moyenne des labels trouvés.

    Parameters
    ----------
    found : Sequence[str]
        Liste des éléments trouvés.
    ground_truth : str
        Vrai label.

    Returns
    -------
    float
        Précision moyenne.
    """
    groups = list(map(lambda index: (found[index], found[: index + 1]), range(len(found))))
    groups = list(filter(lambda group: group[0] == ground_truth, groups))
    precisions = list(map(lambda group: precision(found=group[1], ground_truth=group[0]), groups))
    return fmean(precisions) if precisions else 0.0


def mean_average_precision(retrievals: Sequence[Sequence[str]], labels: Sequence[str]) -> float:
    """
    Calcul de la moyenne des précisions moyennes des labels trouvés.

    Parameters
    ----------
    retrievals : Sequence[Sequence[str]]
        Liste des éléments trouvés.
    labels : Sequence[str]
        Listes des vrais labels.

    Returns
    -------
    float
        Moyenne des précisions moyennes.
    """
    average_precisions = list(
        map(lambda groups: average_precision(found=groups[0], ground_truth=groups[1]), list(zip(retrievals, labels)))
    )
    return fmean(average_precisions)
