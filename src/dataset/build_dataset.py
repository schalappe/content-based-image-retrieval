# -*- coding: utf-8 -*-
"""
Script pour la création des jeux de données.
"""

from glob import glob
from os.path import basename, dirname, join

from polars import DataFrame
from rich.progress import track
from sklearn.model_selection import train_test_split


def create_dataset(input_path: str, output_path: str):
    """
    Création des jeux `train` (base d'images) et `test` (requêtes), stratifiés, ratio 67/33.

    Paths are sorted before the seeded split: `glob` order depends on the file system, and the seed only fixes the
    permutation of positions, not of images.

    Parameters
    ----------
    input_path : str
        Répertoire contenant les données brutes, un sous-répertoire `<couleur>_<style>` par catégorie.
    output_path : str
        Répertoire où stocker les données.
    """
    images_path = sorted(glob(join(input_path, "*", "*.jpg")))
    labels = [basename(dirname(path)) for path in images_path]

    train_set, test_set, train_label, test_label = train_test_split(
        images_path,
        labels,
        test_size=0.33,
        random_state=1331,
        shuffle=True,
        stratify=labels,
    )
    set_info = zip(("train", "test"), ([train_set, train_label], [test_set, test_label]))

    for name, (path, label) in track(set_info, description="Save data ..."):
        color, style = zip(*(item.split("_") for item in label))
        data = DataFrame({"path": path, "label": label, "color": color, "style": style})
        data.write_parquet(join(output_path, f"{name}.parquet"))


if __name__ == "__main__":
    import os
    import sys

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())

    required_vars = ["RAW_PATH", "INPUT_PATH"]
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

    create_dataset(
        input_path=os.environ["RAW_PATH"],
        output_path=os.environ["INPUT_PATH"],
    )
