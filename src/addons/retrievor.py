# -*- coding: utf-8 -*-
"""
Classes pour la recherche d'images similaires.
"""
from abc import ABC, abstractmethod
from typing import Mapping, Dict, List

import numpy as np
from numpy import ndarray
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

from src.addons.extraction.extractor import Extractor


class Finder(ABC):
    """
    Classe générique pour la recherche d'image à partir de similarité entre données caractéristiques.

    Methods
    -------
    search(wanted: str, depth: int = 1)
        Recherche des images similaires dans la base de données.
    """

    def __init__(self, extractor: Extractor, database: Mapping[str, ndarray]):
        self.extractor = extractor
        self.database = database

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

    def search(self, wanted: str, depth: int = 1) -> Dict[str, List[str]]:
        """
        Recherche des images similaires dans la base de données.

        Parameters
        ----------
        wanted : str
            Image à rechercher.
        depth : int
            Le nombre d'images à retourner.

        Returns
        -------
        Dict[str, List[str]]
            Dictionnaire des informations trouvées dans la base.
        """
        # ##: Distance between wanted and database.
        distances = self._compute_distance(self.extractor.extract(image_path=wanted)).flatten()
        nearest_ids = np.argsort(distances)[:depth].tolist()

        # ##: Generate output.
        output = {key: data[nearest_ids].tolist() for key, data in self.database.items() if key != "feature"}
        output.update({"distance": distances[nearest_ids].tolist()})
        return output


class CosinusFinder(Finder):
    def _compute_distance(self, vector: ndarray) -> ndarray:
        return cosine_similarity(self.database["feature"], vector.reshape(1, -1))


class ManhattanFinder(Finder):
    def _compute_distance(self, vector: ndarray) -> ndarray:
        return manhattan_distances(self.database["feature"], vector.reshape(1, -1))


class EuclideanFinder(Finder):
    def _compute_distance(self, vector: ndarray) -> ndarray:
        return euclidean_distances(self.database["feature"], vector.reshape(1, -1))
