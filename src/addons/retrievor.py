# -*- coding: utf-8 -*-
"""
Classes pour la recherche d'images similaires.
"""
from abc import ABC, abstractmethod
from typing import Mapping, Dict, List, Optional

import numpy as np
from numpy import ndarray
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

from src.addons.data import load_database
from src.addons.extraction.extractor import Extractor


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
        distances = self._compute_distance(self.extractor.extract(image_path=wanted)).flatten()
        nearest_ids = np.argsort(distances)[:depth].tolist()

        # ##: Generate output.
        output = {key: data[nearest_ids].tolist() for key, data in self.database.items() if key != "features"}
        output.update({"distance": distances[nearest_ids].tolist()})
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