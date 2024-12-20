import os
from typing import Tuple

import pandas as pd
from Bio import SeqIO
from pypdf import PdfReader


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
