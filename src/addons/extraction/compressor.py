# -*- coding: utf-8 -*-
"""
Ensemble de classe pour l'utilisation de réseaux de neurones pre-entraînés.
"""
from typing import Callable, Optional

import tensorflow as tf
from numpy import ndarray

from src.addons.extraction.extractor import Extractor


class Compressor(Extractor):
    """
    Interface pour l'utilisation d'un réseau de neurones.

    Attributes
    ----------
    extractor : Any
        Object permettant l'extraction des données.
    preprocessor : Callable, default: None
        Fonction de pré-traîtement.

    Methods
    -------
    preprocess(image_path: str)
        Chargement de l'image et ensemble de pré-traitement pour l'extraction des données caractéristiques.
    extract(image_path: str)
        Utilisation d'un réseau de neurones afin d'extraire les données caractéristiques d'une image.
    """

    preprocessor: Optional[Callable]

    def __init__(self, height: int = 224, width: int = 224):
        self.height, self.width = height, width

    def preprocess(self, image_path: str) -> tf.Tensor:
        """
        Chargement de l'image et ensemble de pré-traitement pour l'extraction des données caractéristiques.

        Parameters
        ----------
        image_path : str
            Chemin de l'image.

        Returns
        -------
        tf.Tensor
            L'image prête pour l'extraction des données caractéristiques.
        """
        # ##: Load image.
        image = tf.image.decode_jpeg(tf.io.read_file(image_path))
        if len(tf.shape(image)) < 3:
            image = tf.expand_dims(image, axis=-1)

        # ##: Preprocessing.
        image = tf.image.resize(image, [self.height, self.width])
        image = self.preprocessor(image) if self.preprocessor is not None else image

        return tf.expand_dims(image, axis=1)

    def extract(self, image_path: str) -> ndarray:
        """
        Utilisation d'un réseau de neurones afin d'extraire les données caractéristiques d'une image.

        Parameters
        ----------
        image_path : str
            Chemin de l'image.

        Returns
        -------
        ndarray
            Données caractéristiques de l'image.
        """
        return self.extractor.predict(self.preprocess(image_path=image_path))[0].flatten()


class VGGCompressor(Compressor):
    """
    Interface pour l'utilisation d'un réseau de neurones.

    Attributes
    ----------
    extractor : Any
        Object permettant l'extraction des données.
    preprocessor : Callable, default: None
        Fonction de pré-traîtement.

    Methods
    -------
    preprocess(image_path: str)
        Chargement de l'image et ensemble de pré-traitement pour l'extraction des données caractéristiques.
    extract(image_path: str)
        Utilisation d'un réseau de neurones afin d'extraire les données caractéristiques d'une image.
    """

    def __init__(self, height: int = 224, width: int = 224):
        super().__init__(height=height, width=width)
        self.preprocessor = tf.keras.applications.vgg16.preprocess_input
        model = tf.keras.applications.VGG16(weights="imagenet", include_top=True, input_shape=(height, width, 3))
        self.extractor = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)


class NasNetCompressor(Compressor):
    """
    Interface pour l'utilisation d'un réseau de neurones.

    Attributes
    ----------
    extractor : Any
        Object permettant l'extraction des données.
    preprocessor : Callable, default: None
        Fonction de pré-traîtement.

    Methods
    -------
    preprocess(image_path: str)
        Chargement de l'image et ensemble de pré-traitement pour l'extraction des données caractéristiques.
    extract(image_path: str)
        Utilisation d'un réseau de neurones afin d'extraire les données caractéristiques d'une image.
    """

    def __init__(self, height: int = 224, width: int = 224):
        super().__init__(height=height, width=width)
        self.preprocessor = tf.keras.applications.nasnet.preprocess_input
        model = tf.keras.applications.NASNetLarge(weights="imagenet", include_top=True, input_shape=(height, width, 3))
        self.extractor = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)


class EfficientNetCompressor(Compressor):
    """
    Interface pour l'utilisation d'un réseau de neurones.

    Attributes
    ----------
    extractor : Any
        Object permettant l'extraction des données.
    preprocessor : Callable, default: None
        Fonction de pré-traîtement.

    Methods
    -------
    preprocess(image_path: str)
        Chargement de l'image et ensemble de pré-traitement pour l'extraction des données caractéristiques.
    extract(image_path: str)
        Utilisation d'un réseau de neurones afin d'extraire les données caractéristiques d'une image.
    """

    def __init__(self, height: int = 224, width: int = 224):
        super().__init__(height=height, width=width)
        self.preprocessor = tf.keras.applications.efficientnet.preprocess_input
        model = tf.keras.applications.EfficientNetB7(
            weights="imagenet", include_top=True, input_shape=(height, width, 3)
        )
        self.extractor = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
