import os
import re
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm.auto import tqdm
import pickle as pkl
import itertools

# id to codon and codon to id
id_to_codon = {
    idx: "".join(el)
    for idx, el in enumerate(itertools.product(["A", "T", "C", "G"], repeat=3))
}
codon_to_id = {v: k for k, v in id_to_codon.items()}


def make_dataframe(
    ribo_fname: str, data_path: str, df_trans_to_seq, count_norm: str = "mean"
):
    ribo_fpath = os.path.join(data_path, ribo_fname)

    # Import dataset with ribosome data
    df_ribo = pd.read_csv(
        ribo_fpath,
        sep=" ",
        on_bad_lines="warn",
        dtype=dict(gene="category", transcript="category"),
    ).rename(columns={"count": "counts"})

    # Define count normalization function
    if count_norm == "max":
        f_norm = lambda x: x / x.max()
    elif count_norm == "mean":
        f_norm = lambda x: x / x.mean()
    elif count_norm == "sum":
        f_norm = lambda x: x / x.sum()
    else:
        raise ValueError()

    # Create final dataframe
    final_df = (
        df_ribo.merge(df_trans_to_seq).assign(fname=ribo_fname)
        # Filter spurious positions at the end of the sequence
        .query("position_A_site <= n_codons * 3")
        # Compute normalized counts
        .assign(
            norm_counts=lambda df: df.groupby("gene", observed=True).counts.transform(
                f_norm
            )
        )
    )

    return final_df


def make_all_dataframes(
    data_dirpath: str,
    fa_fpath: str,
    max_n_codons: int = 2000,
    count_norm: str = "mean",
):
    # Import FASTA
    data = []
    with open(fa_fpath, mode="r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            data.append([record.id, str(record.seq)])

    # Create transcripts to sequences mapping

    df_trans_to_seq = pd.DataFrame(data, columns=["transcript", "sequence"])

    # Removes those sequences that have Ns
    sequence_has_n = df_trans_to_seq.sequence.str.contains("N", regex=False)
    df_trans_to_seq = df_trans_to_seq.loc[~sequence_has_n]

    # Number of codons in sequence
    df_trans_to_seq = df_trans_to_seq.assign(
        n_codons=lambda df: df.sequence.str.len() // 3
    )

    # Compute and merge dataframes
    dfs = [
        make_dataframe(
            f,
            df_trans_to_seq=df_trans_to_seq.drop("sequence", axis=1),
            data_path=data_dirpath,
            count_norm=count_norm,
        )
        for f in tqdm(os.listdir(data_dirpath))
        if not f.startswith("ensembl")
    ]
    dfs = pd.concat(dfs)
    for col in ["transcript", "gene", "fname"]:
        dfs[col] = dfs[col].astype("category")

    dfs = dfs.groupby(["transcript", "position_A_site"], observed=True)

    # Average replicates
    dfs = dfs.agg(dict(norm_counts="mean", gene="first")).reset_index()

    dfs = dfs.assign(codon_idx=lambda df: df.position_A_site // 3)
    dfs = dfs.groupby("transcript", observed=True)
    dfs = dfs.agg(
        {
            "norm_counts": lambda x: x.tolist(),
            "codon_idx": lambda x: x.tolist(),
            "gene": "first",
        }
    ).reset_index()
    dfs = dfs.merge(df_trans_to_seq)

    return dfs


def post_process_df(
    df,
    max_consecutive_zeros: int = 30,
    max_n_codons: int = 2000,
    min_coverage: float = 0.5,
):
    df = df.query("n_codons<@max_n_codons")

    codon_sequence = []
    norm_counts = []

    for _, row in df.iterrows():
        seq = row.sequence
        old_nc = row.norm_counts.copy()
        idxs = set(row.codon_idx)
        seq_len = len(seq)

        cs = []
        new_nc = []
        n_zero = 0
        for i in range(seq_len // 3):
            c = i * 3
            cs.append(codon_to_id[seq[c : c + 3]])
            count = np.nan
            if i in idxs:
                count = old_nc.pop(0)
                n_zero = 0
            elif n_zero < max_consecutive_zeros:
                count = 0.0
                n_zero += 1
            new_nc.append(count)
        codon_sequence.append(cs)
        norm_counts.append(new_nc)

    return df.assign(codon_sequence=codon_sequence, norm_counts=norm_counts)


# def fucntion sequence to codon ids
def sequence2codonids(seq):
    codon_ids = []
    for i in range(0, len(seq), 3):
        codon = seq[i : i + 3]
        if len(codon) == 3:
            codon_ids.append(codon_to_id[codon])

    return codon_ids


def process_merged_df(df):
    # remove transcripts with N in sequence
    df = df[df["sequence"].str.contains("N") == False]

    codon_seqs = []
    sequences = list(df["sequence"])
    genes = list(df["gene"])
    transcripts = list(df["transcript"])
    perc_non_zero_annots = []
    norm_counts = list(df["norm_counts"])
    codon_idx = list(df["codon_idx"])
    annot_seqs = []

    for i in range(len(sequences)):
        seq = sequences[i]
        seq = sequence2codonids(seq)
        codon_seqs.append(seq)
        codon_idx_sample = codon_idx[i]
        # convert to list of int
        codon_idx_sample = [int(i) for i in codon_idx_sample[1:-1].split(",")]
        annot_seq_sample = []
        norm_counts_sample = [float(i) for i in norm_counts[i][1:-1].split(",")]
        for j in range(len(seq)):
            if j in codon_idx_sample:
                annot_seq_sample.append(norm_counts_sample[codon_idx_sample.index(j)])
            else:
                annot_seq_sample.append(0.0)
        annot_seqs.append(annot_seq_sample)

        # calculate percentage of non-zero annotations
        perc_non_zero_annots.append(
            sum([1 for i in annot_seq_sample if i != 0.0]) / len(annot_seq_sample)
        )

    final_df = pd.DataFrame(
        list(zip(genes, transcripts, codon_seqs, annot_seqs, perc_non_zero_annots)),
        columns=[
            "gene",
            "transcript",
            "codon_sequence",
            "annotations",
            "perc_non_zero_annots",
        ],
    )

    return final_df
