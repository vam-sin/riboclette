import os

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

from profribo.data.dataset import InstaDeepEmbDataset, RiboDataset
from profribo.data.utils import import_ribo_dataset


class SeqDataModule(LightningDataModule):
    def __init__(
        self,
        min_annot_perc: float = 0.3,
        batch_size: int = 10,
        data_split_seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.min_annot_perc = min_annot_perc
        self.batch_size = batch_size
        self.data_split_seed = data_split_seed

    def setup(self, stage):
        seq_train, seq_test, counts_train, counts_test = import_ribo_dataset(
            os.environ["RIBO_DATA_DIRPATH"], random_state=self.data_split_seed
        )

        self.train_dataset = RiboDataset(
            seq_train, counts_train, min_annot_perc=self.min_annot_perc
        )
        self.test_dataset = RiboDataset(
            seq_test, counts_test, min_annot_perc=self.min_annot_perc
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_dataset)


class InstaDeepDataModule(LightningDataModule):
    def __init__(
        self,
        data_fname: str = "liver_na.hdf5",
        embedding_dim: int = 1280,
        test_perc: float = 0.3,
        data_split_seed: int = 42,
        min_annot_perc: float = 0.3,
        batch_size: int = 10,
        wavelet: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_fname = data_fname
        self.embedding_dim = embedding_dim
        self.test_perc = test_perc
        self.data_split_seed = data_split_seed
        self.min_annot_perc = min_annot_perc
        self.batch_size = batch_size
        self.wavelet = wavelet

    def setup(self, stage):
        full_dataset = InstaDeepEmbDataset(
            hdf5_dset_fpath=os.path.join(
                os.environ["RIBO_DATA_DIRPATH"], self.data_fname
            ),
            min_annot_perc=self.min_annot_perc,
        )
        generator = torch.Generator().manual_seed(self.data_split_seed)
        self.train_dataset, self.test_dataset = random_split(
            full_dataset, [1 - self.test_perc, self.test_perc], generator=generator
        )

        emb_sum, emb_sumsq, emb_n = (
            torch.zeros(self.embedding_dim),
            torch.zeros(self.embedding_dim),
            0,
        )
        for batch, *_ in tqdm(
            DataLoader(self.train_dataset),
            desc="Computing trainset mean and variance...",
        ):
            for seq in batch:
                for e in seq:
                    emb_sum = emb_sum + e
                    emb_sumsq += e * e
                    emb_n += 1

        self.emb_mean = emb_sum / emb_n
        self.emb_var = (emb_sumsq / emb_n) - (self.emb_mean * self.emb_mean)

    def _collate_fn(self, batch, augmentation=False):
        (seq, counts, approx, details) = zip(*list(batch))

        if augmentation:
            tempc = []
            temps = []
            for s, c in zip(seq, counts):
                slen = len(s)
                min_wsize = int(slen * 0.3)
                wsize = torch.randint(min_wsize, slen, (1,)).item()
                wstart = torch.randint(0, slen - wsize + 1, (1,)).item()
                temps.append(s[wstart : wstart + wsize])
                tempc.append(c[wstart : wstart + wsize])
            seq = temps
            counts = tempc

        # Normalize embeddings
        seq = [(s - self.emb_mean) / self.emb_var for s in seq]

        # Compute sequence lengths
        lengths = torch.tensor([len(el) for el in seq])

        # Pad sequences
        seq = nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=0)
        counts = nn.utils.rnn.pad_sequence(counts, batch_first=True, padding_value=0)
        approx = nn.utils.rnn.pad_sequence(approx, batch_first=True, padding_value=0)
        details = nn.utils.rnn.pad_sequence(details, batch_first=True, padding_value=0)

        if self.wavelet:
            return [seq, counts, approx, details, lengths]
        else:
            return [seq, counts, lengths]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=lambda x: self._collate_fn(x, augmentation=False),
            drop_last=True,
            pin_memory=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, collate_fn=lambda x: self._collate_fn(x), batch_size=100
        )
