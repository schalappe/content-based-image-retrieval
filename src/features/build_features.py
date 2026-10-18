# -*- coding: utf-8 -*-
"""
Script pour l'extraction des données caractéristiques des images.
"""

from itertools import batched
from os.path import join
from time import perf_counter
from typing import List, Optional, Sequence

import numpy as np
import polars as pl
from numpy import ndarray
from rich.progress import Progress

from src.addons.extraction.extractor import Extractor, extractors


def gallery_features(extractor: Extractor, paths: Sequence[str], progress: Progress) -> List[Optional[ndarray]]:
    """
    Vecteurs de la base d'images, extraits par lots (non chronométrés).
    """
    task = progress.add_task("  base", total=len(paths))
    features: List[Optional[ndarray]] = []
    for chunk in batched(paths, 256, strict=False):
        features.extend(extractor.extract(chunk))
        progress.advance(task, len(chunk))
    return features


def query_features(extractor: Extractor, paths: Sequence[str], progress: Progress):
    """
    Vecteurs des requêtes, extraits une image à la fois pour chronométrer le coût réel d'une requête.

    The first image is extracted once before timing so that graph tracing and memory allocation are not charged to
    the first query.
    """
    task = progress.add_task("  requêtes", total=len(paths))
    extractor.extract(paths[:1])
    features: List[Optional[ndarray]] = []
    seconds: List[float] = []
    for path in paths:
        begin = perf_counter()
        features.extend(extractor.extract([path]))
        seconds.append(perf_counter() - begin)
        progress.advance(task)
    return features, seconds


def to_column(features: Sequence[Optional[ndarray]]) -> pl.Series:
    """
    Colonne `feature` de taille fixe en float32 ; nulle pour une image sans vecteur.
    """
    dim = next(len(feature) for feature in features if feature is not None)
    matrix = np.zeros((len(features), dim), dtype=np.float32)
    for row, feature in enumerate(features):
        if feature is not None:
            matrix[row] = feature
    missing = pl.Series([feature is None for feature in features])
    return pl.Series("feature", matrix).set(missing, None)


def extract_features(input_path: str, output_path: str):
    """
    Extraction des vecteurs de la base (`train`) et des requêtes (`test`), un fichier `<méthode>.parquet` par méthode.

    Every image keeps its row, with a null `feature` when extraction found nothing, so failures stay countable.
    `extract_seconds` is the single-image extraction time, measured for queries only.

    Parameters
    ----------
    input_path : str
        Répertoire contenant `train.parquet` et `test.parquet`.
    output_path : str
        Répertoire où stocker les vecteurs.
    """
    train = pl.read_parquet(join(input_path, "train.parquet"))
    test = pl.read_parquet(join(input_path, "test.parquet"))

    with Progress() as progress:
        overall = progress.add_task("[green]Extraction des vecteurs ...", total=len(extractors))
        for method, build in extractors.items():
            progress.console.print(f"Méthode {method}")
            extractor = build()
            extractor.fit(train["path"].to_list())
            gallery = gallery_features(extractor, train["path"].to_list(), progress)
            queries, seconds = query_features(extractor, test["path"].to_list(), progress)

            data = pl.concat(
                [
                    train.with_columns(
                        split=pl.lit("train"),
                        extract_seconds=pl.lit(None, dtype=pl.Float64),
                    ),
                    test.with_columns(split=pl.lit("test"), extract_seconds=pl.Series(seconds)),
                ]
            )
            data.with_columns(to_column(gallery + queries)).write_parquet(join(output_path, f"{method}.parquet"))
            progress.advance(overall)


if __name__ == "__main__":
    import os
    import sys

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())

    required_vars = ["INPUT_PATH", "FEATURE_PATH"]
    missing = [var for var in required_vars if not os.environ.get(var, "").strip()]
    if missing:
        print(
            f"Error: Missing required environment variable(s): {', '.join(missing)}\n\n"
            "Please do one of the following:\n"
            "  1. Run 'make prepare' to create the .env file with required variables\n"
            "  2. Manually set the variables in your .env file\n"
            "  3. Export the variables in your shell",
            file=sys.stderr,
        )
        sys.exit(1)

    extract_features(
        input_path=os.environ["INPUT_PATH"],
        output_path=os.environ["FEATURE_PATH"],
    )
