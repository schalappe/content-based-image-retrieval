# -*- coding: utf-8 -*-
"""
Classe générique pour l'extraction des données caractéristiques d'une image.
"""
from abc import ABC, abstractmethod
from typing import Any, Union

from numpy import ndarray
from tensorflow import Tensor

Image = Union[ndarray, Tensor]


class Extractor(ABC):
    """
    Class générique pour l'extraction des données caractéristiques d'une image.

    Attributes
    ----------
    extractor : Any
        Object permettant l'extraction des données.

    Methods
    -------
    preprocess(image_path: str)
        Chargement de l'image et ensemble de pré-traitement pour l'extraction des données caractéristiques.
    extract(image_path: str)
        Extraction des données caractéristiques d'une image.
    """

    extractor: Any

    @abstractmethod
    def preprocess(self, image_path: str) -> Image:
        """
        Chargement de l'image et ensemble de pré-traitement pour l'extraction des données caractéristiques.

        Parameters
        ----------
        image_path : str
            Chemin de l'image.

        Returns
        -------
        Image
            L'image prête pour l'extraction des données caractéristiques.
        """

    @abstractmethod
    def extract(self, image_path: str) -> ndarray:
        """
        Extraction des données caractéristiques d'une image.

        Parameters
        ----------
        image_path : str
            Chemin de l'image.

        Returns
        -------
        ndarray
            Données caractéristiques de l'image.
        """
