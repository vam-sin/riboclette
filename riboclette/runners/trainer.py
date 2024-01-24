import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from jsonargparse import ArgumentParser, namespace_to_dict
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from profribo.data.datamodule import InstaDeepDataModule
from profribo.models.rnn import BiDirGRU
from profribo.models.wrnn import WaveletRNN


class Runner:
    def __init__(
        self,
        model: LightningModule,
        datamodule: LightningDataModule,
        max_epochs: int = 100,
        project_name: str = "ribo_bidirectional",
        patience: int = None,
        run_name: str = "default",
    ):
        self.model = model
        self.datamodule = datamodule
        self.project_name = project_name
        self.max_epochs = max_epochs
        self.patience = patience
        self.run_name = run_name

        self._setup_trainer()

    def _make_callbacks(self):
        callbacks = []

        callbacks.append(ModelCheckpoint(monitor="val_pcc", mode="max", save_last=True))
        if self.patience:
            callbacks.append(
                EarlyStopping(monitor="val_pcc", mode="max", patience=self.patience)
            )

        return callbacks

    def _make_logger(self):
        wandb_logger = WandbLogger(
            project=self.project_name,
            save_dir=os.environ["WANDB_OUTDIR"],
            log_model=False,
            name=self.run_name,
        )

        return wandb_logger

    def _setup_trainer(self):
        self.trainer = Trainer(
            max_epochs=self.max_epochs,
            callbacks=self._make_callbacks(),
            logger=self._make_logger(),
            # gradient_clip_algorithm="norm",
            # gradient_clip_val=1,
        )

    def fit(self):
        self.trainer.fit(model=self.model, datamodule=self.datamodule)


if __name__ == "__main__":
    load_dotenv()

    parser = ArgumentParser()
    parser.add_class_arguments(Runner, skip={"model", "datamodule"})
    parser.add_class_arguments(WaveletRNN, "model")
    parser.add_class_arguments(InstaDeepDataModule, "datamodule")
    cfg = parser.parse_args()

    run_name = datetime.now(timezone(timedelta(hours=+2))).strftime("%y%m%d%H%M%S")
    save_folder = os.path.join(os.environ["CKPTS_PATH"], run_name)
    cfg["run_name"] = run_name

    cfg_w_classes = parser.instantiate_classes(cfg)

    runner = Runner(**cfg_w_classes)
    runner.fit()
