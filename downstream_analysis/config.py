from pyhere import here

TEXTWIDTH_CM = 18
CM_TO_INCH = 1 / 2.54
TEXTWIDTH_INCH = TEXTWIDTH_CM * CM_TO_INCH

GENCODE_FPATH = here("data", "data", "genetic_code.csv")
ENSEMBL_FPATH = here("data", "data", "ensembl.cds.fa")

DISCARDED_TRANSCRIPTS = [
    "ENSMUST00000000828.12",
    "ENSMUST00000060435.6",
    "ENSMUST00000082392.1",
    "ENSMUST00000101801.6",
    "ENSMUST00000115262.7",
    "ENSMUST00000145167.7",
]

ATTR_FNAMES = [
    "241126_RDHPLG_extraLong_int.h5",
    "241001_RDHPLG_test_int.h5",
    "241022_RDHPLG_train_int.h5",
    "241029_RDHPLG_val_int.h5",
    "241105_RDHPLG_extra_int.h5",
]

SAVEFIG_KWARGS = dict(dpi=600, bbox_inches="tight", pad_inches=0.0)
