# -*- coding: utf-8 -*-
"""
Script pour la recherche d'images par similarité sur les vecteurs pré-calculés.
"""

from os.path import join

import polars as pl
from rich.progress import track

from src.addons.extraction.extractor import extractors
from src.addons.finder import MEASURES, search

DEPTH = 5


def inference(input_path: str, feature_path: str, output_path: str):
    """
    Recherche des `DEPTH` plus proches images de chaque requête, pour chaque méthode et chaque mesure.

    Writes `<method>_<measure>_evaluation.parquet`, one row per query. A query without vector gets empty results and
    a gallery image without vector cannot be returned: failures count against the method. `relevant_*` columns hold
    the number of gallery images sharing the query's label, color or style, taken from the full train split so
    every method is scored against the same denominator.

    Parameters
    ----------
    input_path : str
        Répertoire contenant `train.parquet`.
    feature_path : str
        Répertoire contenant les vecteurs `<méthode>.parquet`.
    output_path : str
        Répertoire où stocker les résultats.
    """
    train = pl.read_parquet(join(input_path, "train.parquet"))
    relevant = [
        pl.col(column)
        .replace_strict(
            train[column].value_counts()[column],
            train[column].value_counts()["count"],
            default=0,
        )
        .alias(f"relevant_{column}")
        for column in ("label", "color", "style")
    ]

    combinations = [(method, measure) for method in extractors for measure in MEASURES]
    for method, measure in track(combinations, description="Recherche ..."):
        data = pl.read_parquet(join(feature_path, f"{method}.parquet"))
        gallery = data.filter(pl.col("split") == "train", pl.col("feature").is_not_null())
        queries = data.filter(pl.col("split") == "test").with_columns(relevant)
        found, failed = (
            queries.filter(pl.col("feature").is_not_null()),
            queries.filter(pl.col("feature").is_null()),
        )

        indices, distances, seconds = search(
            gallery["feature"].to_numpy(), found["feature"].to_numpy(), measure, DEPTH
        )
        results = found.with_columns(
            returns=pl.Series(gallery["label"].to_numpy()[indices].tolist()),
            colors=pl.Series(gallery["color"].to_numpy()[indices].tolist()),
            styles=pl.Series(gallery["style"].to_numpy()[indices].tolist()),
            distance=pl.Series(distances.tolist()),
            search_seconds=pl.Series(seconds),
        )
        evaluation = pl.concat([results, failed], how="diagonal_relaxed").with_columns(
            pl.col("returns", "colors", "styles").fill_null([]), depth=pl.lit(DEPTH)
        )
        evaluation.drop("feature", "split").write_parquet(join(output_path, f"{method}_{measure}_evaluation.parquet"))


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
