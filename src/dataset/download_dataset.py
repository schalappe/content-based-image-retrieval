# -*- coding: utf-8 -*-
"""
Script pour télécharger le jeu de données depuis Kaggle.
"""

import shutil
from os.path import exists, isdir, join

import kagglehub
from rich.console import Console

DATASET_ID = "trolukovich/apparel-images-dataset"


def download_dataset(output_path: str) -> str:
    """
    Télécharge le jeu de données Apparel Images depuis Kaggle et le copie vers le répertoire cible.

    Parameters
    ----------
    output_path : str
        Répertoire où copier les données téléchargées.

    Returns
    -------
    str
        Chemin vers le répertoire contenant les données.

    Notes
    -----
    Nécessite une authentification Kaggle configurée. Voir:
    https://github.com/Kaggle/kagglehub?tab=readme-ov-file#authenticate
    """
    console = Console()

    # ##: Download dataset to kagglehub cache directory.
    console.print(f"[bold blue]Downloading dataset:[/bold blue] {DATASET_ID}")
    cache_path = kagglehub.dataset_download(DATASET_ID)
    console.print(f"[green]Downloaded to cache:[/green] {cache_path}")

    # ##!: Validate cache_path exists and is a directory before copying.
    if not exists(cache_path) or not isdir(cache_path):
        raise ValueError(f"Invalid cache path returned: {cache_path}")

    # ##: Check if target directory already contains dataset subdirectories.
    expected_subdirs = ["red_dress", "blue_shirt", "black_pants"]
    if exists(output_path) and any(
        isdir(join(output_path, d)) for d in expected_subdirs
    ):
        console.print(
            f"[yellow]Target directory already contains data:[/yellow] {output_path}"
        )
        console.print(
            "[yellow]Skipping copy. Delete existing data to re-download.[/yellow]"
        )
        return output_path

    console.print(f"[bold blue]Copying to:[/bold blue] {output_path}")
    shutil.copytree(cache_path, output_path, dirs_exist_ok=True)
    console.print("[bold green]Dataset ready![/bold green]")

    return output_path


if __name__ == "__main__":
    import os

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
    download_dataset(output_path=os.environ.get("RAW_PATH"))
