from pyhere import here
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Font sizes
FSS = 5
FSM = 6
FSB = 7

# Figure width
TEXTWIDTH_CM = 18
CM_TO_INCH = 1 / 2.54
TEXTWIDTH_INCH = TEXTWIDTH_CM * CM_TO_INCH

# File paths
GENCODE_FPATH = here("data", "genetic_code.csv")
ENSEMBL_FPATH = here("data", "ensembl.cds.fa")

# File names
ATTR_FNAMES = [
    "241126_RDHPLG_extraLong_int.h5",
    "241001_RDHPLG_test_int.h5",
    "241022_RDHPLG_train_int.h5",
    "241029_RDHPLG_val_int.h5",
    "241105_RDHPLG_extra_int.h5",
]

SPLIT_TO_FNAMES = {
  'train': ["241022_RDHPLG_train_int.h5"],
  'val': ["241029_RDHPLG_val_int.h5"],
  'test': ["241001_RDHPLG_test_int.h5"],
  'int': ["241105_RDHPLG_extra_int.h5"],
}

DISCARDED_TRANSCRIPTS = [
    "ENSMUST00000000828.12",
    "ENSMUST00000060435.6",
    "ENSMUST00000082392.1",
    "ENSMUST00000101801.6",
    "ENSMUST00000115262.7",
    "ENSMUST00000145167.7",
]

SAVEFIG_KWARGS = dict(dpi=600, bbox_inches="tight", pad_inches=0.0)

CONDITIONS_FIXNAME = {
    "CTRL": "CTRL",
    "VAL": "VAL",
    "ILE": "ILE",
    "LEU": "LEU",
    "LEU_ILE": "(L, I)",
    "LEU_ILE_VAL": "(L, I, V)",
}
CONDITIONS_FIXNAME_r = {v: k for k, v in CONDITIONS_FIXNAME.items()}


def rgb_to_rgb01(rgb: tuple[int]):
    return tuple([c / 255 for c in rgb])

id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

AMINO_ACID_MAP = {
    'Val': 'V',  # Valine
    'Ile': 'I',  # Isoleucine
    'Leu': 'L',  # Leucine
    'Lys': 'K',  # Lysine
    'Asn': 'N',  # Asparagine
    'Thr': 'T',  # Threonine
    'Arg': 'R',  # Arginine
    'Ser': 'S',  # Serine
    'Met': 'M',  # Methionine
    'Gln': 'Q',  # Glutamine
    'His': 'H',  # Histidine
    'Pro': 'P',  # Proline
    'Glu': 'E',  # Glutamic acid
    'Asp': 'D',  # Aspartic acid
    'Ala': 'A',  # Alanine
    'Gly': 'G',  # Glycine
    'Tyr': 'Y',  # Tyrosine
    'Cys': 'C',  # Cysteine
    'Trp': 'W',  # Tryptophan
    'Phe': 'F',  # Phenylalanine
}
AMINO_ACID_MAP_r = {v: k for k, v in AMINO_ACID_MAP.items()}

# Colors
ATTR_COL = rgb_to_rgb01((230, 159, 0))
TRUE_COL = rgb_to_rgb01((0, 158, 115))
PRED_COL = rgb_to_rgb01((0, 114, 178))
ASITE_COL = rgb_to_rgb01((213, 94, 0))
DEPRCDN_COL = rgb_to_rgb01((0, 114, 178))

COND_COL = dict(
    CTRL=rgb_to_rgb01((230, 159, 0)),
    VAL=rgb_to_rgb01((0, 114, 178)),
    ILE=rgb_to_rgb01((240, 228, 66)),
    LEU=rgb_to_rgb01((204, 121, 167)),
    LEU_ILE=rgb_to_rgb01((213, 94, 0)),
    LEU_ILE_VAL=rgb_to_rgb01((10, 70, 0)),
)

def amino_acid_cmap():
  colors = np.concatenate((
    plt.get_cmap('tab20b').colors, 
    plt.get_cmap('tab20c').colors))
  np.random.seed(42)
  np.random.shuffle(colors)
  return {a: colors[idx] for idx, a in enumerate(AMINO_ACID_MAP.keys())}
  
AMINO_ACID_COLORS = amino_acid_cmap()
