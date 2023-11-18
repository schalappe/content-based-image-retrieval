# -*- coding: utf-8 -*-
"""
Classe générique pour l'extraction des données caractéristiques d'une image.
"""
from typing import Any, Protocol, Union

from numpy import ndarray
from tensorflow import Tensor

from src.addons.extraction.compressor import (
    EfficientNetCompressor,
    NasNetCompressor,
    VGGCompressor,
)
from src.addons.extraction.descriptor import (
    AKAZEDescriptor,
    ORBDescriptor,
    SIFTDescriptor,
)

Image = Union[ndarray, Tensor]


class Extractor(Protocol):
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


extractors = {
    "AKAZE": AKAZEDescriptor,
    "ORB": ORBDescriptor,
    "SIFT": SIFTDescriptor,
    "VGG": VGGCompressor,
    "NasNet": NasNetCompressor,
    "EfficientNet": EfficientNetCompressor,
}
