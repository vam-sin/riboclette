from pyhere import here

# Font sizes
FSS = 5
FSM = 6
FSB = 7

# Figure width
TEXTWIDTH_CM = 18
CM_TO_INCH = 1 / 2.54
TEXTWIDTH_INCH = TEXTWIDTH_CM * CM_TO_INCH

# File paths
GENCODE_FPATH = here("data", "data", "genetic_code.csv")
ENSEMBL_FPATH = here("data", "data", "ensembl.cds.fa")

# File names
ATTR_FNAMES = [
    "241126_RDHPLG_extraLong_int.h5",
    "241001_RDHPLG_test_int.h5",
    "241022_RDHPLG_train_int.h5",
    "241029_RDHPLG_val_int.h5",
    "241105_RDHPLG_extra_int.h5",
]

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
        'CTRL': 'CTRL',
        'VAL': 'VAL',
        'ILE': 'ILE',
        'LEU': 'LEU',
        'LEU_ILE': '(L, I)', 
        'LEU_ILE_VAL': '(L, I, V)'}

def rgb_to_rgb01(rgb: tuple[int]):
    return tuple([c / 255 for c in rgb])

# Colors
ATTR_COL = rgb_to_rgb01((230,159,0))
TRUE_COL = rgb_to_rgb01((0,114,178))
PRED_COL = rgb_to_rgb01((0,158,115))
ASITE_COL = rgb_to_rgb01((213,94,0))
DEPRCDN_COL = rgb_to_rgb01((0,114,178))

COND_COL = dict(
  CTRL=rgb_to_rgb01((230,159,0)),
  VAL=rgb_to_rgb01((0,114,178)),
  ILE=rgb_to_rgb01((240,228,66)),
  LEU=rgb_to_rgb01((204,121,167)),
  LIV=rgb_to_rgb01((213,94,0)),
  LI=rgb_to_rgb01((10,70,0,0))
)

