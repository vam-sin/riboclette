import h5py
import numpy as np
import pywt
import torch
from torch.utils.data import Dataset


class RiboDataset(Dataset):
    def __init__(self, sequences, counts, mlm_proba=None, min_annot_perc=None):
        self.sequences = sequences
        self.counts = counts

        if min_annot_perc:
            good_idxs = [
                idx
                for idx, el in enumerate(self.counts)
                if (np.array(el).astype(float) != 0).mean() >= min_annot_perc
            ]
            self.counts = [self.counts[idx] for idx in good_idxs]
            self.sequences = [self.sequences[idx] for idx in good_idxs]

        self.mlm_proba = mlm_proba

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = np.array(self.sequences[idx]).astype(int)
        counts = np.log1p(np.array(self.counts[idx]).astype(float))
        if self.mlm_proba:
            sequence *= np.random.binomial(1, self.mlm_proba, sequence.shape[0])
        return torch.IntTensor(sequence), torch.FloatTensor(counts)


class InstaDeepEmbDataset(Dataset):
    def __init__(self, hdf5_dset_fpath: str, mlm_proba=None, min_annot_perc=None):
        data = h5py.File(hdf5_dset_fpath, "r")
        self.embs = data["embeddings"]
        self.counts = data["counts6"]

        self.min_annot_perc = min_annot_perc

        if self.min_annot_perc:
            self.good_idxs = [
                idx
                for idx in range(self.counts.len())
                if ((self.counts[idx] != 0) & (~np.isnan(self.counts[idx])))[
                    15:-15
                ].mean()
                >= self.min_annot_perc
            ]

        self.mlm_proba = mlm_proba

    def __len__(self):
        if self.min_annot_perc:
            return len(self.good_idxs)
        else:
            return len(self.embs.len())

    def __getitem__(self, idx):
        if self.min_annot_perc:
            idx = self.good_idxs[idx]

        counts = np.log1p(self.counts[idx])

        wavelet = "haar"
        coeffs = pywt.dwt(np.nan_to_num(counts), wavelet)
        approx = pywt.idwt(coeffs[0], None, wavelet)[: len(counts)]
        approx[np.isnan(counts)] = np.nan

        details = pywt.idwt(None, coeffs[1], wavelet)[: len(counts)]
        details[np.isnan(counts)] = np.nan

        emb = self.embs[idx].reshape(len(counts), -1)
        if self.mlm_proba:
            sequence *= np.random.binomial(1, self.mlm_proba, emb.shape[0])
        return (
            torch.FloatTensor(emb),
            torch.FloatTensor(counts),
            torch.FloatTensor(approx),
            torch.FloatTensor(details),
        )
