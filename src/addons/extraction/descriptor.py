# -*- coding: utf-8 -*-
"""
Ensemble de classe pour l'utilisation des descripteurs du module `OpenCV`.
"""
from typing import Optional

import cv2 as cv
from numpy import ndarray, concatenate, zeros, array

from src.addons.extraction.extractor import Extractor


class Descriptor(Extractor):
    """
    Interface pour l'utilisation des descripteurs du module `OpenCV`.

    Attributes
    ----------
    extractor : Any
        Object permettant l'extraction des données.
    vector_size : int
        Taille du vecteur des données caractéristiques.

    Methods
    -------
    preprocess(image_path: str)
        Chargement de l'image et ensemble de pré-traitement pour l'extraction des données caractéristiques.
    extract(image_path: str)
        Utilisation d'un descripteur afin d'extraire les données caractéristiques d'une image.
    """

    def __init__(self, size: int = 32):
        self.vector_size = size

    def preprocess(self, image_path: str) -> ndarray:
        """
        Chargement de l'image et ensemble de pré-traitement pour l'extraction des données caractéristiques.

        Parameters
        ----------
        image_path : str
            Chemin de l'image.

        Returns
        -------
        ndarray
            L'image prête pour l'extraction des données caractéristiques.
        """
        return cv.imread(image_path)

    def extract(self, image_path: str) -> Optional[ndarray]:
        """
        Utilisation d'un descripteur afin d'extraire les données caractéristiques d'une image.

        Parameters
        ----------
        image_path : str
            Chemin de l'image.

        Returns
        -------
        ndarray
            Données caractéristiques de l'image.
        """
        # ##: Get image key points and keep the best.
        image = self.preprocess(image_path=image_path)
        kps = self.extractor.detect(image)
        if not kps:
            return None
        kps = sorted(kps, key=lambda x: -x.response)[: self.vector_size]

        # ##: Computing descriptors vector
        kps, dsc = self.extractor.compute(image, kps)
        dsc = dsc.flatten()

        # ##: Vector size adjustment.
        needed_size = self.vector_size * 64
        if dsc.size < needed_size:
            dsc = concatenate([dsc, zeros(needed_size - dsc.size)])
        return array(dsc)


class AKAZEDescriptor(Descriptor):
    """
    Interface pour l'utilisation du descripteur **AKAZE** du module `OpenCV`.

    Attributes
    ----------
    extractor : Any
        Object permettant l'extraction des données.
    vector_size : int
        Taille du vecteur des données caractéristiques.

    Methods
    -------
    preprocess(image_path: str)
        Chargement de l'image et ensemble de pré-traitement pour l'extraction des données caractéristiques.
    extract(image_path: str)
        Utilisation d'un descripteur afin d'extraire les données caractéristiques d'une image.
    """

    def __init__(self, size: int = 32):
        super().__init__(size=size)
        self.extractor = cv.AKAZE_create()


class ORBDescriptor(Descriptor):
    """
    Interface pour l'utilisation du descripteur **ORB** du module `OpenCV`.

    Attributes
    ----------
    extractor : Any
        Object permettant l'extraction des données.
    vector_size : int
        Taille du vecteur des données caractéristiques.

    Methods
    -------
    preprocess(image_path: str)
        Chargement de l'image et ensemble de pré-traitement pour l'extraction des données caractéristiques.
    extract(image_path: str)
        Utilisation d'un descripteur afin d'extraire les données caractéristiques d'une image.
    """

    def __init__(self, size: int = 32):
        super().__init__(size=size)
        self.extractor = cv.ORB_create()


class SURFDescriptor(Descriptor):
    """
    Interface pour l'utilisation du descripteur **SURF** du module `OpenCV`.

    Attributes
    ----------
    extractor : Any
        Object permettant l'extraction des données.
    vector_size : int
        Taille du vecteur des données caractéristiques.

    Methods
    -------
    preprocess(image_path: str)
        Chargement de l'image et ensemble de pré-traitement pour l'extraction des données caractéristiques.
    extract(image_path: str)
        Utilisation d'un descripteur afin d'extraire les données caractéristiques d'une image.
    """

    def __init__(self, size: int = 32):
        super().__init__(size=size)
        self.extractor = cv.xfeatures2d.SURF_create(400)
