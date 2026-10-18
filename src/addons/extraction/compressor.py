# -*- coding: utf-8 -*-
"""
Réseaux de neurones pré-entraînés (timm) utilisés comme extracteurs.
"""

from typing import List, Sequence

import timm
import torch
from numpy import ndarray
from PIL import Image

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Compressor:
    """
    Vecteur global d'image d'un réseau pré-entraîné : sortie de `timm` sans tête de classification.

    Each checkpoint uses its own prescribed preprocessing (size, crop, normalisation) from its timm configuration.
    `extract` works on any number of images: the gallery is processed in batches, a query is a batch of one.

    Parameters
    ----------
    name : str
        Nom du modèle pré-entraîné dans `timm`.
    batch_size : int, default: 4
        Nombre d'images par passe; small batches avoid memory pressure on Apple GPUs with the largest networks.
    """

    def __init__(self, name: str, batch_size: int = 4):
        self.model = timm.create_model(name, pretrained=True, num_classes=0).eval().to(DEVICE)
        self.transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=self.model))
        self.batch_size = batch_size

    def fit(self, image_paths: Sequence[str]):
        """Rien à apprendre : les poids sont figés."""

    def load(self, image_path: str) -> torch.Tensor:
        """
        Lecture en RGB (y compris les images en niveaux de gris) et pré-traitement du modèle.
        """
        with Image.open(image_path) as image:
            return self.transform(image.convert("RGB"))

    @torch.inference_mode()
    def extract(self, image_paths: Sequence[str]) -> List[ndarray]:
        """
        Vecteur caractéristique de chaque image.
        """
        features: List[ndarray] = []
        for start in range(0, len(image_paths), self.batch_size):
            batch = torch.stack([self.load(path) for path in image_paths[start : start + self.batch_size]])
            features.extend(self.model(batch.to(DEVICE)).float().cpu().numpy())
        return features


def vgg16() -> Compressor:
    """VGG16, couche dense `fc7` avant la classification (4096), comme l'étude d'origine."""
    return Compressor("vgg16.tv_in1k")


def nasnet() -> Compressor:
    """NASNetLarge, moyenne globale (4032)."""
    return Compressor("nasnetalarge.tf_in1k")


def efficientnet() -> Compressor:
    """EfficientNet-B7, moyenne globale (2560), poids ImageNet AutoAugment, ceux de Keras."""
    return Compressor("tf_efficientnet_b7.aa_in1k")


def convnext() -> Compressor:
    """ConvNeXt-Tiny, CNN moderne de taille modeste, moyenne globale (768)."""
    return Compressor("convnext_tiny.fb_in1k")


def dinov2() -> Compressor:
    """DINOv2 ViT-B/14, auto-supervisé, jeton de classe (768)."""
    return Compressor("vit_base_patch14_dinov2.lvd142m")


def siglip2() -> Compressor:
    """SigLIP 2 ViT-B/16, pré-entraîné sur des paires image-texte, encodeur d'image (768)."""
    return Compressor("vit_base_patch16_siglip_224.v2_webli")
