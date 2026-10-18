# -*- coding: utf-8 -*-
"""
Sac de mots visuels construit sur des descripteurs locaux binaires d'`OpenCV` (ORB, AKAZE).
"""

from typing import Any, List, Optional, Sequence

import cv2 as cv
import numpy as np
from numpy import ndarray
from sklearn.cluster import MiniBatchKMeans


class BagOfWords:
    """
    Histogramme de mots visuels d'une image, de taille fixe `words`.

    Local descriptors have no fixed count or order, so they cannot be compared position by position. Each descriptor
    is assigned to its nearest visual word and the image becomes the normalised word histogram, which every
    similarity measure can compare. Call `fit` on the train split only, then `extract`.

    The vocabulary is k-means on unpacked descriptor bits with centroids rounded back to bits, so assignment uses
    the Hamming distance, the native distance of binary descriptors.
    """

    def __init__(
        self,
        detector: Any,
        words: int = 512,
        sample_images: int = 2000,
        per_image: int = 100,
        seed: int = 1331,
    ):
        self.detector = detector
        self.words, self.sample_images, self.per_image, self.seed = (
            words,
            sample_images,
            per_image,
            seed,
        )
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING)
        self.vocabulary: Optional[ndarray] = None

    def descriptors(self, image_path: str) -> Optional[ndarray]:
        """
        Descripteurs binaires de l'image, `None` si aucun point clé n'est détecté.
        """
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image illisible : {image_path}")
        _, descriptors = self.detector.detectAndCompute(image, None)
        return descriptors

    def fit(self, image_paths: Sequence[str]):
        """
        Apprentissage du vocabulaire sur un échantillon reproductible des images.
        """
        rng = np.random.default_rng(self.seed)
        sample = rng.choice(
            len(image_paths),
            size=min(self.sample_images, len(image_paths)),
            replace=False,
        )
        collected = []
        for index in sample:
            descriptors = self.descriptors(image_paths[index])
            if descriptors is not None:
                keep = rng.choice(
                    len(descriptors),
                    size=min(self.per_image, len(descriptors)),
                    replace=False,
                )
                collected.append(descriptors[keep])
        bits = np.unpackbits(np.concatenate(collected), axis=1).astype(np.float32)
        kmeans = MiniBatchKMeans(n_clusters=self.words, random_state=self.seed, n_init=3).fit(bits)
        self.vocabulary = np.packbits(kmeans.cluster_centers_ > 0.5, axis=1)

    def extract(self, image_paths: Sequence[str]) -> List[Optional[ndarray]]:
        """
        Histogramme normalisé de chaque image, `None` pour une image sans point clé.
        """
        if self.vocabulary is None:
            raise RuntimeError("Appeler `fit` avant `extract`.")
        features: List[Optional[ndarray]] = []
        for image_path in image_paths:
            descriptors = self.descriptors(image_path)
            if descriptors is None:
                features.append(None)
                continue
            words = [match.trainIdx for match in self.matcher.match(descriptors, self.vocabulary)]
            histogram = np.bincount(words, minlength=self.words).astype(np.float32)
            features.append(histogram / histogram.sum())
        return features


def akaze() -> BagOfWords:
    """AKAZE avec son descripteur binaire MLDB par défaut."""
    return BagOfWords(cv.xfeatures2d.AKAZE_create())


def orb() -> BagOfWords:
    """ORB, limité par défaut à 500 points clés par image."""
    return BagOfWords(cv.ORB_create())
