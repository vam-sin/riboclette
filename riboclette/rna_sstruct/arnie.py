import numpy as np
from arnie.bpps import bpps
from arnie.mea.mea import MEA
import pandas as pd
from tqdm.auto import tqdm


def get_nucleotide_pairing_proba(df, package="vienna_2"):
    return df.assign(
        bpp=lambda df: [
            np.sum(bpps(seq, package=package), axis=0) for seq in tqdm(df.sequence)
        ]
    )


def get_maximum_expected_accuracy_seq(
    df: pd.DataFrame,
    log_gamma_range: tuple[int, int] = (-4, 4),
    package: str = "vienna_2",
):
    results = []
    for seq in tqdm(df.sequence):
        output = _get_maximum_expected_accuracy_seq(
            seq, log_gamma_range=log_gamma_range, package=package
        )
        results.append(output + (seq,))
    results = pd.DataFrame(
        results, columns=["bpp", "mcc", "mea_struct", "log_gamma", "sequence"]
    )

    return df.merge(results, on="sequence")


def _get_maximum_expected_accuracy_seq(
    seq: str, log_gamma_range: tuple[int, int], package: str
):
    bp_matrix = bpps(seq, package=package)

    scores = []
    structures = []
    log_gammas = []
    for log_gamma in range(*log_gamma_range):
        mea_mdl = MEA(bp_matrix, gamma=10**log_gamma)

        log_gammas.append(log_gamma)
        structures.append(mea_mdl.structure)

        # Matthews's correlation coefficient
        mcc = mea_mdl.score_expected()[2]
        scores.append(mcc)

    print(scores)
    best_mcc_idx = np.argmax(scores)

    return (
        np.sum(bp_matrix, axis=0),
        scores[best_mcc_idx],
        structures[best_mcc_idx],
        log_gammas[best_mcc_idx],
    )
