import itertools
import os
import re

import config
import h5py
import numpy as np
import pandas as pd
from Bio import SeqIO
from pyhere import here
from tqdm import tqdm, trange

fnames = [
    "241126_RDHPLG_extraLong_int.h5",
    "241001_RDHPLG_test_int.h5",
    "241022_RDHPLG_train_int.h5",
    "241029_RDHPLG_val_int.h5",
    "241105_RDHPLG_extra_int.h5",
]


def _global_attr_plot(f, num_samples, ctrl: bool = True, topk: int = 5) -> list:
    attr_top5 = []

    for i in trange(num_samples):
        if f["transcript"][i] in config.DISCARDED_TRANSCRIPTS:
            continue

        num_codons_sample = len(f["x_input"][i]) - 1
        dname = "lig_ctrl" if ctrl else "lig_dd"
        lig_sample_ctrl = f[dname][i].reshape(num_codons_sample, num_codons_sample)

        for j in range(num_codons_sample):
            lig_sample_ctrl[j] = lig_sample_ctrl[j] / np.sum(np.abs(lig_sample_ctrl[j]))
            # take absolute value of lig_sample_ctrl[j]
            lig_sample_ctrl[j] = np.abs(lig_sample_ctrl[j])
            # get top 5 codons with highest lig_sample_ctrl[j]
            top5 = np.argsort(lig_sample_ctrl[j])[-topk:]
            # get distance between top 5 codons and the codon of interest
            for k in range(topk):
                attr_top5.append(top5[k] - j)
    return attr_top5


def global_attr_plot(ctrl: bool, test=False, topk: int = 5):
    output = []
    for fname in fnames:
        with h5py.File(here("data", "results", "interpretability", fname), "r") as f:
            num_samples = 1 if test else len(f["x_input"])

            temp_output = _global_attr_plot(f, num_samples, ctrl=ctrl, topk=topk)
            output.extend(temp_output)
    if test:
        print(output)
    else:
        np.save(
            here("data", "results", "plotting", f"globl_attr_plot_{ctrl}"),
            arr=output,
        )


def run_global_attr_plot():
    global_attr_plot(ctrl=True)
    global_attr_plot(ctrl=False)


def run_global_stalling(window_size: int = 20):

    condition_values = {
        "CTRL": 64,
        "ILE": 65,
        "LEU": 66,
        "LEU_ILE": 67,
        "LEU_ILE_VAL": 68,
        "VAL": 69,
    }
    condition_values_inverse = {
        64: "CTRL",
        65: "ILE",
        66: "LEU",
        67: "LEU_ILE",
        68: "LEU_ILE_VAL",
        69: "VAL",
    }
    # global variables
    id_to_codon = {
        idx: "".join(el)
        for idx, el in enumerate(itertools.product(["A", "T", "C", "G"], repeat=3))
    }
    codon_to_id = {v: k for k, v in id_to_codon.items()}

    condition_codon_stall = {
        "CTRL": {codon: [] for codon in id_to_codon.values()},
        "ILE": {codon: [] for codon in id_to_codon.values()},
        "LEU": {codon: [] for codon in id_to_codon.values()},
        "LEU_ILE": {codon: [] for codon in id_to_codon.values()},
        "LEU_ILE_VAL": {codon: [] for codon in id_to_codon.values()},
        "VAL": {codon: [] for codon in id_to_codon.values()},
    }

    stop_codons = ["TAA", "TAG", "TGA"]
    for condition in condition_values.keys():
        for codon in stop_codons:
            condition_codon_stall[condition].pop(codon)

    for fname in fnames:
        f = h5py.File(here("data", "results", "interpretability", fname), "r")

        num_samples = len(f["condition"])

        for i in tqdm(range(num_samples)):

            if f["transcript"][i] in config.DISCARDED_TRANSCRIPTS:
                continue

            sample_condition = f["condition"][i].decode("utf-8")
            if sample_condition == "CTRL":
                y_true_full_sample = f["y_true_full"][i]
            else:
                y_true_full_sample = f["y_true_dd"][i]
            x_input_sample = f["x_input"][i][1:]
            y_true_full_sample_norm = y_true_full_sample / np.nanmax(y_true_full_sample)
            for j in range(len(y_true_full_sample_norm)):
                if (
                    np.isnan(y_true_full_sample[j]) == False
                    and id_to_codon[int(x_input_sample[j])]
                    in condition_codon_stall[sample_condition]
                    and y_true_full_sample[j] != 0.0
                ):
                    condition_codon_stall[sample_condition][
                        id_to_codon[int(x_input_sample[j])]
                    ].append(y_true_full_sample_norm[j])

        f.close()

    condition_codon_stall_mean = {
        condition: {
            codon: np.mean(condition_codon_stall[condition][codon])
            for codon in condition_codon_stall[condition]
        }
        for condition in condition_values.keys()
    }
    # sort the dictionary by the mean stall value in descending order
    condition_codon_stall_mean_sorted = {
        condition: {
            k: v
            for k, v in sorted(
                condition_codon_stall_mean[condition].items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        for condition in condition_values.keys()
    }

    pd.DataFrame.from_dict(condition_codon_stall_mean_sorted).to_csv(
        here("data", "results", "plotting", "condition_codon_stall_mean_sorted.zip")
    )

    condition_codon_attr_peaks = {
        "CTRL": {codon: [] for codon in id_to_codon.values()},
        "ILE": {codon: [] for codon in id_to_codon.values()},
        "LEU": {codon: [] for codon in id_to_codon.values()},
        "LEU_ILE": {codon: [] for codon in id_to_codon.values()},
        "LEU_ILE_VAL": {codon: [] for codon in id_to_codon.values()},
        "VAL": {codon: [] for codon in id_to_codon.values()},
    }
    condition_codon_attr_full = {
        "CTRL": {codon: [] for codon in id_to_codon.values()},
        "ILE": {codon: [] for codon in id_to_codon.values()},
        "LEU": {codon: [] for codon in id_to_codon.values()},
        "LEU_ILE": {codon: [] for codon in id_to_codon.values()},
        "LEU_ILE_VAL": {codon: [] for codon in id_to_codon.values()},
        "VAL": {codon: [] for codon in id_to_codon.values()},
    }

    stop_codons = ["TAA", "TAG", "TGA"]
    for condition in condition_values.keys():
        for codon in stop_codons:
            condition_codon_attr_peaks[condition].pop(codon)
            condition_codon_attr_full[condition].pop(codon)

    for fname in fnames:
        f = h5py.File(here("data", "results", "interpretability", fname), "r")

        num_samples = len(f["condition"])
        for i in tqdm(range(num_samples)):

            if f["transcript"][i] in config.DISCARDED_TRANSCRIPTS:
                continue

            sample_cond = f["condition"][i].decode("utf-8")
            x_input_sample = f["x_input"][i][1:]
            num_codons = len(x_input_sample)
            y_true_full_sample = f["y_true_full"][i]
            if sample_cond == "CTRL":
                lig_attr_ctrl_sample = f["lig_ctrl"][i].reshape(num_codons, num_codons)
            else:
                lig_attr_ctrl_sample = f["lig_dd"][i].reshape(num_codons, num_codons)

            # find the indices of the codons with the top 10 highest values
            top10_indices = np.argsort(-y_true_full_sample)[:10]
            # set j to be starting points, and k to be the end points
            for j in range(len(y_true_full_sample)):
                # a_site = top10_indices[j]
                a_site = j
                start = a_site - window_size
                end = a_site + window_size + 1

                lig_attr_ctrl_sample_window = lig_attr_ctrl_sample[a_site][start:end]
                if len(lig_attr_ctrl_sample_window) == (window_size * 2) + 1:
                    lig_attr_ctrl_sample_window_norm = (
                        lig_attr_ctrl_sample_window
                        / np.max(np.abs(lig_attr_ctrl_sample_window))
                    )
                    x_input_sample_window = x_input_sample[start:end]
                    for l in range(len(x_input_sample_window)):
                        if (
                            id_to_codon[int(x_input_sample_window[l])]
                            in condition_codon_attr_full[sample_cond]
                        ):
                            condition_codon_attr_full[sample_cond][
                                id_to_codon[int(x_input_sample_window[l])]
                            ].append(lig_attr_ctrl_sample_window_norm[l])

            for j in range(len(top10_indices)):
                a_site = top10_indices[j]
                start = a_site - window_size
                end = a_site + window_size + 1

                lig_attr_ctrl_sample_window = lig_attr_ctrl_sample[a_site][start:end]
                if len(lig_attr_ctrl_sample_window) == (window_size * 2) + 1:
                    lig_attr_ctrl_sample_window_norm = (
                        lig_attr_ctrl_sample_window
                        / np.max(np.abs(lig_attr_ctrl_sample_window))
                    )
                    x_input_sample_window = x_input_sample[start:end]
                    for l in range(len(x_input_sample_window)):
                        if (
                            id_to_codon[int(x_input_sample_window[l])]
                            in condition_codon_attr_peaks[sample_cond]
                        ):
                            condition_codon_attr_peaks[sample_cond][
                                id_to_codon[int(x_input_sample_window[l])]
                            ].append(lig_attr_ctrl_sample_window_norm[l])

        f.close()

    condition_codon_attr_full_mean = {
        condition: {
            codon: np.mean(condition_codon_attr_full[condition][codon])
            for codon in condition_codon_attr_full[condition]
        }
        for condition in condition_values.keys()
    }
    # sort the dictionary by the mean stall value in descending order
    condition_codon_attr_full_mean_sorted = {
        condition: {
            k: v
            for k, v in sorted(
                condition_codon_attr_full_mean[condition].items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        for condition in condition_values.keys()
    }

    pd.DataFrame.from_dict(condition_codon_attr_full_mean_sorted).to_csv(
        here("data", "results", "plotting", "condition_codon_attr_full_mean_sorted.zip")
    )

    condition_codon_attr_peaks_mean = {
        condition: {
            codon: np.mean(condition_codon_attr_peaks[condition][codon])
            for codon in condition_codon_attr_peaks[condition]
        }
        for condition in condition_values.keys()
    }
    # sort the dictionary by the mean stall value in descending order
    condition_codon_attr_peaks_mean_sorted = {
        condition: {
            k: v
            for k, v in sorted(
                condition_codon_attr_peaks_mean[condition].items(),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        for condition in condition_values.keys()
    }

    pd.DataFrame.from_dict(condition_codon_attr_peaks_mean_sorted).to_csv(
        here(
            "data", "results", "plotting", "condition_codon_attr_peaks_mean_sorted.zip"
        )
    )


def run_topk_attr_condition_wise(
    wsize: int = 20,
    attr_type: str = "lig",
):

    # Load genetic code
    genetic_code = pd.read_csv(
        here("data", "data", "genetic_code.csv"), index_col=0
    ).set_index("Codon")
    genetic_code.head()

    # Load gene to seq dataframe
    df_trans_to_seq = []
    with open(here("data", "data", "ensembl.cds.fa"), mode="r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            df_trans_to_seq.append(
                [
                    record.id,
                    str(record.seq),
                    record.description.split("gene_symbol:")[1].split()[0],
                ]
            )
    df_trans_to_seq = pd.DataFrame(
        df_trans_to_seq, columns=["transcript", "sequence", "symbol"]
    )

    # Define conditions
    conditions = ["ILE", "LEU", "LEU_ILE", "LEU_ILE_VAL", "VAL", "CTRL"]

    # Store Weldford datastructure for each combination of codon and condition
    condition_freq_depr_head = {
        cond: {cod: [] for cod in genetic_code.index}
        for cond in conditions
        if cond != "CTRL"
    }

    for fname in fnames:

        f = h5py.File(here("data", "results", "interpretability", fname), "r")
        transcripts = f["transcript"][:].astype("U")
        conditions = f["condition"][:].astype("U")

        # Loop throught transcripts
        for transc_idx in trange(transcripts.shape[0]):
            condition = conditions[transc_idx]

            # Exclude control condition
            if condition == "CTRL":
                continue

            transcript = transcripts[transc_idx]

            if transcript in config.DISCARDED_TRANSCRIPTS:
                continue
                print(transcript)

            # Get attribution vector and reshape to matrix
            trasc_attr = f[f"{attr_type}_dd"][transc_idx]
            n_codons = int(np.sqrt(trasc_attr.shape[0]))
            trasc_attr = trasc_attr.reshape(n_codons, n_codons)

            # Get sequence
            sequence = df_trans_to_seq.query(
                "transcript == @transcript"
            ).sequence.values[0]
            sequence = np.array(re.findall("...", sequence))

            depr_true = f["y_true_dd"][transc_idx]
            # good_idxs = np.argsort(depr_true)[-20:]
            good_idxs = np.nonzero(depr_true > np.mean(depr_true) + np.std(depr_true))[
                0
            ]
            good_idxs = good_idxs[
                (good_idxs >= wsize) & (good_idxs < n_codons - wsize - 1)
            ]
            # threshold = np.nanquantile(np.abs(depr_true), .9)
            # good_idxs = np.nonzero(np.abs(depr_true) > threshold)[0]
            # good_idxs = good_idxs[(good_idxs>=wsize) & (good_idxs < n_codons-wsize-1)]

            # Loop through the sequence, excluding a prefix and suffix of wsize and wsize+1, respectively
            for idx in good_idxs:  # np.arange(wsize, n_codons-wsize-1):
                # for idx in np.arange(n_codons-1):
                # wattr = trasc_attr[idx]
                wattr = trasc_attr[idx, idx - wsize : idx + wsize + 1]
                wattr = wattr / np.abs(wattr).max()
                for attr_idx, codon_idx in enumerate(
                    np.arange(idx - wsize, idx + wsize + 1)
                ):
                    # for attr_idx, codon_idx in enumerate(np.arange(n_codons-1)):
                    condition_freq_depr_head[condition][sequence[codon_idx]].append(
                        wattr[attr_idx]
                    )
        f.close()

    condition_freq_depr_head = {
        cond: {cod: np.mean(w) for cod, w in cod_wise.items()}
        for cond, cod_wise in condition_freq_depr_head.items()
    }
    condition_freq_depr_head = pd.DataFrame(condition_freq_depr_head).to_csv(
        here("data", "result", "plotting", "topk_attr_condition_wise.zip")
    )


if __name__ == "__main__":
    print("Running global_attr_plot")
    run_global_attr_plot()

    # print("Running global_stalling")
    # run_global_stalling()

    # print("Running make_topk_attr_condition_wise")
    # run_topk_attr_condition_wise()
