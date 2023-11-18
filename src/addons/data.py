# -*- coding: utf-8 -*-
"""
Module pour le chargement des données.
"""
import polars as pl
from numpy import ndarray, asarray
from typing import Mapping


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
        "features": asarray(data.select(pl.col("feature")).to_series().to_list()),
        "colors": asarray(data.select(pl.col("color")).to_series().to_list()),
        "styles": asarray(data.select(pl.col("style")).to_series().to_list()),
    }
