# -*- coding: utf-8 -*-
"""
Script pour la recherche d'image par similarité.
"""
from copy import deepcopy
from itertools import product
from os.path import join

import polars as pl

from src.addons.data import load_database
from src.addons.extraction.extractor import extractors
from src.addons.finder import CosinusFinder, EuclideanFinder, Finder, ManhattanFinder

finders = {"cosinus": CosinusFinder, "euclidean": EuclideanFinder, "manhattan": ManhattanFinder}


def inference(input_path: str, feature_path: str, output_path: str):
    """
    Élaboration et enregistrement de prédictions.

    Parameters
    ----------
    input_path : str
        Répertoire contenant les données brutes.
    feature_path : str
        Répertoire contenant les données caractéristiques.
    output_path : str
        Répertoire où stocker les prédictions.
    """
    # ##: Get data.
    data = pl.read_parquet(join(input_path, "test.parquet"))

    # ##: Loop over finder and extractors.
    for [(extract_method, extract_func), (finder_method, finder_func)] in product(extractors.items(), finders.items()):
        finder: Finder = finder_func(extract_func(), load_database(join(feature_path, f"{extract_method}_db.parquet")))

        # ##: Make predictions.
        prediction = []
        for content in data.to_dicts():
            res = deepcopy(content)
            found = finder.search(wanted=res["path"], depth=5)
            res.update(found)
            prediction.append(res)

        # ##: Store.
        prediction = pl.DataFrame(prediction)
        prediction.write_parquet(join(output_path, f"{extract_method}_{finder_method}_evaluation.parquet"))


if __name__ == "__main__":
    import os

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
    inference(
        input_path=os.environ.get("INPUT_PATH"),
        feature_path=os.environ.get("FEATURE_PATH"),
        output_path=os.environ.get("EVALUATION_PATH"),
    )
