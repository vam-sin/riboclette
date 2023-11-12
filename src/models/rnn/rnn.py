import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection

from riboclette.utils.loss import *
from riboclette.utils.metrics import BinaryMaskedMetric, CorrCoef


class BiDirGRU(LightningModule):
    def __init__(
        self,
        embedding_dsize: int = 100,
        embedding_dim: int = 1280,
        gru_hsize: int = 300,
        gru_n_layers: int = 2,
        gru_dropout: float = 0.3,
        comb_max_duration: int = 100,
        lr: float = 1e-3,
        weight_decay=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embedding_dsize = embedding_dsize
        self.embedding_dim = embedding_dim
        self.gru_hsize = gru_hsize
        self.gru_n_layers = gru_n_layers
        self.gru_dropout = gru_dropout
        self.lr = lr
        self.weight_decay = weight_decay

        self.loss_func = MaskedCombinedPearsonLoss(comb_max_duration=comb_max_duration)

        metrics = MetricCollection(dict(pcc=CorrCoef()))
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")

        self.define_modules()

    def define_modules(self):
        self.emb = nn.Embedding(
            num_embeddings=5**3 + 1,
            embedding_dim=self.embedding_dim,
            padding_idx=0,
        )
        gru_input_size = self.embedding_dim

        # self.cnn = nn.Conv1d(self.embedding_dim, 200, 1)
        # gru_input_size = 100

        self.rnn = nn.GRU(
            input_size=gru_input_size,
            hidden_size=self.gru_hsize,
            num_layers=self.gru_n_layers,
            dropout=self.gru_dropout,
            bidirectional=True,
            batch_first=True,
        )

        # self.linear = torch.nn.Linear(self.gru_hsize * 2, 1)

    def update_metric(
        self,
        batch,
        outputs,
        metric,
    ):
        X, y_true, lengths = batch
        y_pred, _ = outputs

        mask = torch.arange(X.shape[1])[None, :].to(lengths) < lengths[:, None].to(X)
        mask = torch.logical_and(mask, torch.logical_not(torch.isnan(y_true)))

        metric.update(y_pred, y_true, mask)

    def forward(self, X, lengths):
        # X = self.emb(X)
        # X = torch.swapaxes(X, 1, 2)

        # X = self.cnn(X)

        # X = torch.swapaxes(X, 1, 2)

        X = nn.utils.rnn.pack_padded_sequence(
            X, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.rnn(X)
        outputs, lengths = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # return nn.functional.softplus(self.linear(outputs))
        return nn.functional.softplus(outputs.sum(dim=2))

    def eval_forward(self, batch, outputs=None):
        return self.forward(batch)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def _step(self, batch, batch_idx):
        X, y_true, y_approx, y_details, lengths = batch

        y_pred = self(X, lengths)

        # Mask based on leghts
        mask = torch.arange(y_pred.shape[1])[None, :].to(lengths) < lengths[:, None]
        mask = torch.logical_and(mask, torch.logical_not(torch.isnan(y_true)))

        loss = self.loss_func(y_pred, y_true, mask, timestamp=self.current_epoch)

        stage = self.trainer.state.stage
        if stage in ["train", "sanity_check"]:
            metrics = self.train_metrics(y_pred, y_true, mask)
            self.log_dict(metrics)
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        elif stage == "validate":
            self.valid_metrics.update(y_pred, y_true, mask)
        else:
            raise ValueError(f"Unknown stage {stage}")

        return loss

    def on_validation_epoch_end(self):
        output = self.valid_metrics.compute()
        self.log_dict(output)
        self.valid_metrics.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
