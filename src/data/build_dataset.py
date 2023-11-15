# -*- coding: utf-8 -*-
"""
Script pour la création des jeux de données.
"""
from glob import glob
from os.path import sep, join

from polars import DataFrame
from rich.progress import track
from sklearn.model_selection import train_test_split


def create_dataset(input_path: str, output_path: str):
    """
    Création des jeux de données `train_set` et `test_set` avec un ratio de 70/30.

    Parameters
    ----------
    input_path : str
        Répertoire contenant les données brutes
    output_path : str
        Répertoire où stocker les données.
    """
    # ##: Get necessary.
    images_path = glob(input_path + "*/*.jpg")
    labels = list(map(lambda item: item.split(sep)[-2], images_path))

    # ##: Split and save.
    train_set, train_label, test_set, test_label = train_test_split(
        images_path, labels, test_size=0.33, random_state=1331, shuffle=True, stratify=labels
    )
    set_info = zip(("train", "test"), ([train_set, train_label], [test_set, test_label]))

    for name, path, label in track(set_info, description="Save data ..."):
        data = DataFrame({"label": label, "path": path})
        data.write_parquet(join(output_path, f"{name}.parquet"))


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())
    create_dataset(input_path=os.environ.get("RAW_DATA"), output_path=os.environ.get("INPUT_DATA"))
