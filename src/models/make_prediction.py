# -*- coding: utf-8 -*-
"""
Script pour la recherche d'image par similarité.
"""
from itertools import product
from os.path import join

import polars as pl
from rich.progress import Progress

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
    with Progress() as progress:
        combinations = list(product(extractors.items(), finders.items()))

        tasks = progress.add_task("[green]Réalisation des prédictions ...", total=len(combinations))
        for [(extract_method, extract_func), (finder_method, finder_func)] in combinations:
            task = progress.add_task(f"[red]Prédiction avec {extract_method} et {finder_method}", total=data.shape[0])
            finder: Finder = finder_func(
                extract_func(), load_database(join(feature_path, f"{extract_method}_db.parquet"))
            )

            # ##: Make predictions.
            prediction = []
            for content in data.to_dicts():
                color, style = content["label"].split("_")
                res = {"ground_truth": content["label"], "gt_color": color, "gt_style": style}
                res.update(finder.search(wanted=content["path"], depth=5))

                prediction.append(res)
                progress.advance(task)

            # ##: Store.
            prediction = pl.DataFrame(prediction)
            prediction.write_parquet(join(output_path, f"{extract_method}_{finder_method}_evaluation.parquet"))
            progress.advance(tasks)


if __name__ == "__main__":
    import os
    import sys

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())

    required_vars = ["INPUT_PATH", "FEATURE_PATH", "EVALUATION_PATH"]
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

    inference(
        input_path=os.environ["INPUT_PATH"],
        feature_path=os.environ["FEATURE_PATH"],
        output_path=os.environ["EVALUATION_PATH"],
    )
