import os
from typing import Tuple

import pandas as pd
from Bio import SeqIO
from pypdf import PdfReader
from contextlib import contextmanager
import matplotlib.pyplot as plt


def get_pdf_dimensions_in_cm(file_path: str | os.PathLike) -> Tuple[float, float]:
    reader = PdfReader(file_path)

    # Get the first page
    first_page = reader.pages[0]

    # Get dimensions in points
    media_box = first_page.mediabox
    width_points = float(media_box.width)
    height_points = float(media_box.height)

    # Convert points to centimeters
    width_cm = width_points * 2.54 / 72
    height_cm = height_points * 2.54 / 72

    return width_cm, height_cm


def read_ensembl(fa_fpath: str | os.PathLike) -> pd.DataFrame:
    data = []
    with open(fa_fpath, mode="r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            data.append(
                [
                    record.id,
                    str(record.seq),
                    record.description.split("gene_symbol:")[1].split()[0],
                ]
            )

    df = pd.DataFrame(data, columns=["transcript", "sequence", "symbol"])

    return df


@contextmanager
def journal_plotting_ctx():
    """
    A context manager to temporarily apply specific plotting parameters for matplotlib.
    """
    # Save the current rcParams
    original_rc = plt.rcParams.copy()

    # Define the desired parameters
    SMALL_SIZE = 5
    MEDIUM_SIZE = 6
    BIGGER_SIZE = 7

    try:
        # Apply the desired rc settings
        plt.rc(
            "font", size=SMALL_SIZE, family="arial"
        )  # Default font sizes and family
        plt.rc("axes", titlesize=BIGGER_SIZE, labelsize=SMALL_SIZE)  # Axes sizes
        plt.rc("xtick", labelsize=SMALL_SIZE)  # Tick label sizes
        plt.rc("ytick", labelsize=SMALL_SIZE)
        plt.rc(
            "legend", fontsize=SMALL_SIZE, title_fontsize=BIGGER_SIZE
        )  # Legend sizes
        plt.rc("figure", titlesize=BIGGER_SIZE)  # Figure title size
        plt.rcParams["svg.fonttype"] = "none"  # SVG fonttype

        # Yield control to the block of code
        yield
    finally:
        # Restore original rcParams
        plt.rcParams.update(original_rc)
    return contextmanager, journal_plotting_ctx
