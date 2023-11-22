# -*- coding: utf-8 -*-
"""
Classes pour la recherche d'images similaires.
"""
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np
from numpy import ndarray
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)

from src.addons.data import load_database
from src.addons.extraction.extractor import Extractor


def timeit(func: Callable):
    """
    Ajout de la durée d'exécution dans la réponse.
    Ce décorateur fonctionne uniquement si la sortie de la fonction à enrichir `func` est un dictionnaire.

    Parameters
    ----------
    func : Callable
        Fonction à enrichir.

    Returns
    -------
    Callable
        Fonction enrichie.
    """

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        result.update({"duration": end_time - start_time})
        return result

    return timeit_wrapper


class Finder(ABC):
    """
    Classe générique pour la recherche d'image à partir de similarité entre données caractéristiques.

    Methods
    -------
    change_database(data_path: str)
        Changement de la base des données caractéristiques.
    search(wanted: str, depth: int = 1)
        Recherche des images similaires dans la base de données.
    """

    def __init__(self, extractor: Extractor, database: Optional[Mapping[str, ndarray]] = None):
        self.extractor = extractor
        self.database = database

    def change_database(self, data_path: str):
        """
        Changement de la base des données caractéristiques.

        Parameters
        ----------
        data_path : str
            Chemin des données à charger.
        """
        self.database = load_database(data_path=data_path)

    @abstractmethod
    def _compute_distance(self, vector: ndarray) -> ndarray:
        """
        Calcul de la distance entre le vecteur caractéristique et les éléments de la base de données.

        Parameters
        ----------
        vector : ndarray
            Vecteur caractéristique

        Returns
        -------
        ndarray
            Distance avec les éléments la base de données.
        """

    @timeit
    def search(self, wanted: str, depth: int = 1) -> Dict[str, List[str]]:
        """
        Recherche des images similaires dans la base de données.

        Parameters
        ----------
        wanted : str
            Image à rechercher.
        depth : int, default: 1
            Le nombre d'images à retourner.

        Returns
        -------
        Dict[str, List[str]]
            Dictionnaire des informations trouvées dans la base.
        """
        if self.database is None:
            raise RuntimeError("Aucune base de données n'a été fournie.")

        # ##: Distance between wanted and database.
        vector = self.extractor.extract(image_path=wanted)
        if vector is not None:
            distances = self._compute_distance(vector).flatten()
            nearest_ids = np.argsort(distances)[:depth].tolist()

            # ##: Generate output.
            color = self.database["colors"][nearest_ids].tolist()
            style = self.database["styles"][nearest_ids].tolist()
            distance = distances[nearest_ids].tolist()
        else:
            distance, color, style = [], [], []

        predicts = list(map(lambda item: "_".join(item), zip(color, style)))

        output = {"input": wanted, "colors": color, "styles": style, "returns": predicts, "distance": distance}
        return output


class CosinusFinder(Finder):
    """
    Recherche d'images similaire à partir de la distance de Cosinus.
    """

    def _compute_distance(self, vector: ndarray) -> ndarray:
        """
        Calcul de la distance entre le vecteur caractéristique et les éléments de la base de données.

        Parameters
        ----------
        vector : ndarray
            Vecteur caractéristique

        Returns
        -------
        ndarray
            Distance avec les éléments la base de données.
        """
        return cosine_similarity(self.database["features"], vector.reshape(1, -1))


class ManhattanFinder(Finder):
    """
    Recherche d'images similaire à partir de la distance de Manhattan.
    """

    def _compute_distance(self, vector: ndarray) -> ndarray:
        """
        Calcul de la distance entre le vecteur caractéristique et les éléments de la base de données.

        Parameters
        ----------
        vector : ndarray
            Vecteur caractéristique

        Returns
        -------
        ndarray
            Distance avec les éléments la base de données.
        """
        return manhattan_distances(self.database["features"], vector.reshape(1, -1))


class EuclideanFinder(Finder):
    """
    Recherche d'images similaire à partir de la distance Euclidienne.
    """

    def _compute_distance(self, vector: ndarray) -> ndarray:
        """
        Calcul de la distance entre le vecteur caractéristique et les éléments de la base de données.

        Parameters
        ----------
        vector : ndarray
            Vecteur caractéristique

        Returns
        -------
        ndarray
            Distance avec les éléments la base de données.
        """
        return euclidean_distances(self.database["features"], vector.reshape(1, -1))
