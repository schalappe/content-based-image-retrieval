# -*- coding: utf-8 -*-
"""
Réseaux de neurones pré-entraînés sur ImageNet utilisés comme extracteurs.
"""

from typing import Callable, List, Sequence

import tensorflow as tf
from numpy import ndarray


class Compressor:
    """
    Vecteur de la couche choisie d'un réseau pré-entraîné.

    `extract` works on any number of images: the gallery is processed in batches, a query is a batch of one.

    Parameters
    ----------
    model : tf.keras.Model
        Réseau dont la sortie est le vecteur caractéristique.
    size : int
        Côté de l'image attendu par le réseau.
    preprocess : Callable
        Pré-traitement propre au réseau, appliqué à une image RGB en [0, 255].
    batch_size : int, default: 4
        Nombre d'images par passe. Measured on an M1 Pro (Metal) with EfficientNetB7 at 600 px: 4 images is ~2x
        faster per image than 1, while 16 is ~7x slower (memory pressure).
    """

    def __init__(
        self,
        model: tf.keras.Model,
        size: int,
        preprocess: Callable,
        batch_size: int = 4,
    ):
        self.model, self.size, self.preprocess, self.batch_size = (
            model,
            size,
            preprocess,
            batch_size,
        )

    def fit(self, image_paths: Sequence[str]):
        """Rien à apprendre : les poids sont figés."""

    def load(self, image_path: tf.Tensor) -> tf.Tensor:
        """
        Lecture en RGB (y compris les images en niveaux de gris), redimensionnement et pré-traitement.
        """
        image = tf.io.decode_image(tf.io.read_file(image_path), channels=3, expand_animations=False)
        image = tf.image.resize(image, [self.size, self.size])
        return self.preprocess(image)

    def extract(self, image_paths: Sequence[str]) -> List[ndarray]:
        """
        Vecteur caractéristique de chaque image.
        """
        images = tf.data.Dataset.from_tensor_slices(list(image_paths)).map(self.load).batch(self.batch_size)
        return [vector for batch in images for vector in self.model(batch, training=False).numpy()]


def vgg16() -> Compressor:
    """VGG16, couche dense `fc2` (4096), celle de l'étude d'origine."""
    base = tf.keras.applications.VGG16(weights="imagenet", include_top=True)
    model = tf.keras.Model(inputs=base.input, outputs=base.get_layer("fc2").output)
    return Compressor(model, 224, tf.keras.applications.vgg16.preprocess_input)


def nasnet() -> Compressor:
    """NASNetLarge, moyenne globale de la dernière carte de caractéristiques (4032)."""
    model = tf.keras.applications.NASNetLarge(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(331, 331, 3)
    )
    return Compressor(model, 331, tf.keras.applications.nasnet.preprocess_input)


def efficientnet() -> Compressor:
    """EfficientNetB7, moyenne globale de la dernière carte de caractéristiques (2560); normalisation intégrée."""
    model = tf.keras.applications.EfficientNetB7(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(600, 600, 3)
    )
    return Compressor(model, 600, tf.keras.applications.efficientnet.preprocess_input)
