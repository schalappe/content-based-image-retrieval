# -*- coding: utf-8 -*-
"""
Module pour le chargement des données.
"""
from typing import Mapping

import polars as pl
from numpy import ndarray


def load_database(data_path: str) -> Mapping[str, ndarray]:
    """
    Changement des données caractéristiques des images.

    Parameters
    ----------
    data_path : str
        Chemin des données à charger.

    Returns
    -------
    Mapping[str, ndarray]
        Dictionnaires des données et des labels.
    """
    data = pl.read_parquet(data_path)
    return {
        "features": data.select(pl.col("feature")).to_numpy(),
        "colors": data.select(pl.col("color")).to_numpy(),
        "styles": data.select(pl.col("style")).to_numpy(),
    }
