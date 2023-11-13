# -*- coding: utf-8 -*-
"""
Script pour l'extraction des données caractéristiques des images.
"""
from os.path import join

import polars as pl
from polars import DataFrame
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from src.addons.extraction.compressor import VGGCompressor, NasNetCompressor, EfficientNetCompressor
from src.addons.extraction.descriptor import AKAZEDescriptor, ORBDescriptor, SURFDescriptor


def extract_features(input_path: str, output_path: str):
    """
    Extraction des données caractéristiques afin de constituer les bases de données.

    Parameters
    ----------
    input_path : str
        Répertoire contenant les données.
    output_path : str
        Répertoire où stocker les bases de données.
    """
    # ##: Prepare necessary.
    data = pl.read_parquet(join(input_path, "train.parquet"))
    extractors = {
        "AKAZE": AKAZEDescriptor(),
        "ORB": ORBDescriptor(),
        "SURF": SURFDescriptor(),
        "VGG": VGGCompressor(),
        "NasNet": NasNetCompressor(),
        "EfficientNet": EfficientNetCompressor(),
    }

    # ##: Build database.
    with Progress("", BarColumn(), "", TimeElapsedColumn(), TimeRemainingColumn()) as progress:
        overall_task = progress.add_task("[green]Création des base de données ...", total=len(extractors))
        for method, extractor in extractors.items():
            extract_task = progress.add_task(f"Extraction avec la méthode {method}", total=data.shape[0])

            # ##: Build database.
            database = []
            for content in data.to_dicts():
                feature = extractor.extract(image_path=content["path"])
                color, style = content["label"].split("_")
                database.append({"feature": feature, "color": color, "style": style})
                progress.advance(extract_task)

            # ##: Save database.
            database = DataFrame(database)
            database.write_parquet(join(output_path, f"{method}_db.parquet"))
            progress.advance(overall_task)
