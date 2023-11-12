import os
import csv
import pickle as pkl
from sklearn.model_selection import train_test_split


def import_ribo_dataset(
    ribo_data_dirpath: str, train_size: float = 0.7, random_state: int = 42
):
    with open(os.path.join(ribo_data_dirpath, "raw_counts.csv"), "r") as read_obj:
        counts = list(csv.reader(read_obj))
    with open(os.path.join(ribo_data_dirpath, "raw_sequences.csv"), "r") as read_obj:
        sequences = list(csv.reader(read_obj))

    return train_test_split(
        sequences, counts, train_size=train_size, random_state=random_state
    )


def import_v_data():
    with open(
        os.path.join(os.environ["V_DATA_DIRPATH"], "CTRL_train_feats.pkl"), "rb"
    ) as fname:
        train = pkl.load(fname)
    with open(
        os.path.join(os.environ["V_DATA_DIRPATH"], "CTRL_val_feats.pkl"), "rb"
    ) as fname:
        test = pkl.load(fname)

    seq_train, seq_test, counts_train, counts_test = [], [], [], []
    for _, val in train.items():
        seq_train.append(val[0])
        counts_train.append(val[1])

    for _, val in test.items():
        seq_test.append(val[0])
        counts_test.append(val[1])

    return seq_train, seq_test, counts_train, counts_test
