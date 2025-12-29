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
    import sys

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())

    raw_path = os.environ.get("RAW_PATH", "").strip()
    if not raw_path:
        print(
            "Error: RAW_PATH environment variable is not set or is empty.\n\n"
            "Please do one of the following:\n"
            "  1. Run 'make prepare' to create the .env file with required variables\n"
            "  2. Manually set RAW_PATH in your .env file (e.g., RAW_PATH=data/raw)\n"
            "  3. Export RAW_PATH in your shell: export RAW_PATH=data/raw",
            file=sys.stderr,
        )
        sys.exit(1)

    download_dataset(output_path=raw_path)
