# -*- coding: utf-8 -*-
"""
Contrat commun des extracteurs et registre des méthodes comparées.
"""

from typing import Callable, Dict, List, Optional, Protocol, Sequence

from numpy import ndarray

from src.addons.extraction.compressor import efficientnet, nasnet, vgg16
from src.addons.extraction.descriptor import akaze, orb


class Extractor(Protocol):
    """
    Extraction de vecteurs de taille fixe.

    `fit` learns from the train split only (no-op for pretrained networks). `extract` returns one vector per image,
    or `None` when the method finds nothing to describe (e.g. no keypoint).
    """

    def fit(self, image_paths: Sequence[str]): ...

    def extract(self, image_paths: Sequence[str]) -> List[Optional[ndarray]]: ...


extractors: Dict[str, Callable[[], Extractor]] = {
    "AKAZE": akaze,
    "ORB": orb,
    "VGG": vgg16,
    "NasNet": nasnet,
    "EfficientNet": efficientnet,
}
