# libraries
import numpy as np
import pandas as pd 
import torch
import random
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torchmetrics.functional import pearson_corrcoef
from torchmetrics.regression import MeanAbsolutePercentageError, MeanAbsoluteError
from torchmetrics import Metric
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import lightning as L
import itertools
import os

# global variables
id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

# dataset generation functions
def longestZeroSeqLength(a):
    '''
    length of the longest sub-sequence of zeros
    '''
    a = a[1:-1].split(', ')
    a = [float(k) for k in a]
    # longest sequence of zeros
    longest = 0
    current = 0
    for i in a:
        if i == 0.0:
            current += 1
        else:
            longest = max(longest, current)
            current = 0
    return longest

def percNans(a):
    '''
    returns the percentage of nans in the sequence
    '''
    a = a[1:-1].split(', ')
    a = [float(k) for k in a]
    a = np.asarray(a)
    perc = np.count_nonzero(np.isnan(a)) / len(a)

    return perc

def RiboDatasetGWS(data_folder: str, dataset_split: str, ds: str, threshold: float = 0.6, longZerosThresh: int = 20, percNansThresh: float = 0.1):
    if ds == 'ALL':
        # # paths 
        # ctrl_path = data_folder + 'CTRL.csv'
        # leu_path = data_folder + 'LEU.csv'
        # arg_path = data_folder + 'ARG.csv'

        # # load the data
        # df_ctrl = pd.read_csv(ctrl_path)
        # df_ctrl['condition'] = 'CTRL'
        # df_leu = pd.read_csv(leu_path)
        # df_leu['condition'] = 'LEU'
        # df_arg = pd.read_csv(arg_path)
        # df_arg['condition'] = 'ARG'

        # df_full = pd.concat([df_ctrl, df_leu, df_arg], axis=0) # ctrl, leu, arg

        # # drop first column
        # df_full = df_full.drop(['Unnamed: 0'], axis=1)

        # df_full.columns = ['gene', 'transcript', 'sequence', 'annotations', 'perc_non_zero_annots', 'condition']

        # # apply annot threshold
        # df_full = df_full[df_full['perc_non_zero_annots'] >= threshold]

        # # for all the sequences in a condition that is not CTRL, add their respective CTRL sequence to them
        # sequences_ctrl = []
        # annotations_list = list(df_full['annotations'])
        # condition_df_list = list(df_full['condition'])
        # transcripts_list = list(df_full['transcript'])

        # for i in range(len(condition_df_list)):
        #     try:
        #         if condition_df_list[i] != 'CTRL':
        #             # find the respective CTRL sequence for the transcript
        #             ctrl_sequence = df_full[(df_full['transcript'] == transcripts_list[i]) & (df_full['condition'] == 'CTRL')]['annotations'].iloc[0]
        #             sequences_ctrl.append(ctrl_sequence)
        #         else:
        #             sequences_ctrl.append(annotations_list[i])
        #     except:
        #         sequences_ctrl.append('NA')

        # # add the sequences_ctrl to the df
        # print(len(sequences_ctrl), len(annotations_list))
        # df_full['ctrl_sequence'] = sequences_ctrl

        # # remove those rows where the ctrl_sequence is NA
        # df_full = df_full[df_full['ctrl_sequence'] != 'NA']

        # # sanity check for the ctrl sequences
        # # get the ds with only condition as CTRL
        # df_ctrl_full = df_full[df_full['condition'] == 'CTRL']
        # ctrl_sequences_san = list(df_ctrl_full['annotations'])
        # ctrl_sequences_san2 = list(df_ctrl_full['ctrl_sequence'])

        # for i in range(len(ctrl_sequences_san)):
        #     assert ctrl_sequences_san[i] == ctrl_sequences_san2[i]

        # print("Sanity Checked")

        # # get longest zero sequence length for each sequence in annotations and ctrl_sequence
        # annotations_list = list(df_full['annotations'])
        # sequences_ctrl = list(df_full['ctrl_sequence'])
        # annotation_long_zeros = []
        # ctrl_sequence_long_zeros = []
        # num_nans_full = []
        # num_nans_ctrl = []
        # for i in range(len(annotations_list)):
        #     annotation_long_zeros.append(longestZeroSeqLength(annotations_list[i]))
        #     ctrl_sequence_long_zeros.append(longestZeroSeqLength(sequences_ctrl[i]))
        #     num_nans_full.append(percNans(annotations_list[i]))
        #     num_nans_ctrl.append(percNans(sequences_ctrl[i]))

        # # add the longest zero sequence length to the df
        # df_full['longest_zero_seq_length_annotation'] = annotation_long_zeros
        # df_full['longest_zero_seq_length_ctrl_sequence'] = ctrl_sequence_long_zeros

        # # add the number of nans to the df
        # df_full['perc_nans_annotation'] = num_nans_full
        # df_full['perc_nans_ctrl_sequence'] = num_nans_ctrl

        # # apply the threshold for the longest zero sequence length
        # df_full = df_full[df_full['longest_zero_seq_length_annotation'] <= longZerosThresh]
        # df_full = df_full[df_full['longest_zero_seq_length_ctrl_sequence'] <= longZerosThresh]

        # # apply the threshold for the number of nans
        # df_full = df_full[df_full['perc_nans_annotation'] <= percNansThresh]
        # df_full = df_full[df_full['perc_nans_ctrl_sequence'] <= percNansThresh]

        # # GWS
        # genes = df_full['gene'].unique()
        # genes_train, genes_test = train_test_split(genes, test_size=0.2, random_state=42)

        # # split the dataframe
        # df_train = df_full[df_full['gene'].isin(genes_train)]
        # df_test = df_full[df_full['gene'].isin(genes_test)]

        out_train_path = 'data/dh/ribo_train_' + str(dataset_split) + '-NA_dh_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '.csv'
        out_test_path = 'data/dh/ribo_test_' + str(dataset_split) + '-NA_dh_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '.csv'

        # df_train.to_csv(out_train_path, index=False)
        # df_test.to_csv(out_test_path, index=False)

        df_train = pd.read_csv(out_train_path)
        df_test = pd.read_csv(out_test_path)

        return df_train, df_test

class GWSDatasetFromPandasVAE(Dataset):
    def __init__(self, df):
        self.df = df
        self.counts = list(self.df['annotations'])
        self.sequences = list(self.df['sequence'])
        self.condition_lists = list(self.df['condition'])
        self.condition_values = {'CTRL': 0, 'LEU': 1, 'ARG': 2}
        self.max_len = 10000

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        X = self.df['sequence'].iloc[idx]
        # convert to int
        X = X[1:-1].split(', ')
        X = [int(i) for i in X]

        y = self.df['annotations'].iloc[idx]
        # convert string into list of floats
        y = y[1:-1].split(', ')
        y = [float(i) for i in y]

        y = [1+i for i in y]
        y = np.log(y)

        # ctrl sequence 
        ctrl_y = self.df['ctrl_sequence'].iloc[idx]
        # convert string into list of floats
        ctrl_y = ctrl_y[1:-1].split(', ')
        ctrl_y = [float(i) for i in ctrl_y]

        # no min max scaling
        ctrl_y = [1+i for i in ctrl_y]
        ctrl_y = np.log(ctrl_y)

        X = np.array(X)
        # multiply X with condition value times 64 + 1
        add_factor = (self.condition_values[self.condition_lists[idx]]) * 64
        X += add_factor

        y = np.array(y)
        len_X = len(X)

        X = torch.from_numpy(X).long()
        y = torch.from_numpy(y).float()
        ctrl_y = torch.from_numpy(ctrl_y).float()

        gene = self.df['gene'].iloc[idx]
        transcript = self.df['transcript'].iloc[idx]

        # pad y with 0s
        y_pad = torch.zeros((self.max_len))
        y_pad[:len_X] = y
        
        return y_pad, len_X

class LSTMEncoder(nn.Module):
    def __init__(self, latent_dims, hidden_dim, num_layers, bidirectional, bs):
        super(LSTMEncoder, self).__init__()

        self.latent_dims = latent_dims

        self.lstm_e1 = nn.LSTM(1, hidden_dim, batch_first=True, bidirectional=True)

        # mean and variance for latent space
        if bidirectional:
            self.fc_mu = nn.Linear(hidden_dim * 2, latent_dims)
            self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dims)
        else:
            self.fc_mu = nn.Linear(hidden_dim, latent_dims)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dims)

        self.bs = bs
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.training = True

    def forward(self, x, lengths):

        dev = torch.device("cpu")
        lengths = lengths.to(dev)

        x = torch.nan_to_num(x, nan=0.0, posinf=None, neginf=None)

        max_len = torch.max(lengths)

        # clip x to the size of lengths
        x = x[:,:max_len]

        x = torch.unsqueeze(x, 2)
        x, hid = self.lstm_e1(x)
        # print shapes of all the tensors in hid
        if self.bidirectional:
            hid = hid[0].reshape(-1, self.hidden_dim * 2)
        else:
            hid = hid[0].reshape(-1, self.hidden_dim)

        hid = F.relu(hid)

        out_mu = self.fc_mu(hid)
        out_logvar = self.fc_logvar(hid)

        return out_mu, out_logvar, lengths

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dims, hidden_dim, n_layers, bidirectional, bs):
        super(LSTMDecoder, self).__init__()
        self.latent_dims = latent_dims
        self.lstm_d1 = nn.LSTM(latent_dims, hidden_dim, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden_dim, latent_dims)
        if bidirectional:
            self.fc1 = nn.Linear(hidden_dim * 2, 1)
        else:
            self.fc1 = nn.Linear(hidden_dim, 1)
        self.n_layers = n_layers
        self.bs = bs
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

    def forward(self, decoder_input, lengths):
        h_0 = torch.autograd.Variable(torch.zeros(self.n_layers * 2 if self.bidirectional else self.n_layers, self.bs, self.hidden_dim).cuda()) # (1, bs, hidden)
        c_0 = torch.autograd.Variable(torch.zeros(self.n_layers * 2 if self.bidirectional else self.n_layers, self.bs, self.hidden_dim).cuda()) # (1, bs, hidden)

        max_len = torch.max(lengths)

        decoder_input = decoder_input.unsqueeze(1)

        decoder_outputs = []
        decoder_hidden = (h_0, c_0)

        for i in range(max_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_hidden[0][-1, :, :].unsqueeze(0).permute(1, 0, 2)
            decoder_input = self.out(decoder_input)

        x = torch.cat(decoder_outputs, dim=1)
        x = F.relu(x)
        x = self.fc1(x)
        return x

    def forward_step(self, x, hid):
        x, hid = self.lstm_d1(x, hid)
        return x, hid

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, lengths):

        max_len = torch.max(lengths)
        y_pred = torch.squeeze(y_pred, 2)
        y_true = y_true[:,:max_len]
        # lengths
        mask = torch.arange(y_true.shape[1])[None, :].to(lengths) < lengths[:, None]
        mask = torch.logical_and(mask, torch.logical_not(torch.isnan(y_true)))

        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        # assert y_pred_mask and y_true_mask dont have nans
        assert torch.isnan(y_pred_mask).any() == False
        assert torch.isnan(y_true_mask).any() == False

        loss = nn.functional.l1_loss(y_pred_mask, y_true_mask, reduction="none")

        return torch.sqrt(loss.mean())

class VAE(L.LightningModule):
    def __init__(self, latent_dim: int, num_batches: int, num_epochs: int, hidden_dims: int, num_layers: int, bidirectional: bool, bs: int, lr: float):
        super().__init__()
        # # Saving hyperparameters of autoencoder
        # self.save_hyperparameters()
        # # Creating encoder and decoder
        self.encoder = LSTMEncoder(latent_dim, hidden_dims, num_layers, bidirectional, bs)
        self.decoder = LSTMDecoder(latent_dim, hidden_dims, num_layers, bidirectional, bs)
        self.loss_l1 = MaskedL1Loss()
        self.total_batches = num_batches
        self.batches_per_epoch = num_batches / num_epochs
        self.num_epochs = num_epochs
        self.lr = lr

    def reparameterization(self, mean, var):
        device = mean.device
        epsilon = torch.randn_like(var).to(device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick

        return z

    def forward(self, x, lens):
        # set nans in x to 0
        
        mean, log_var, seq_len = self.encoder(x, lens)
        
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)

        x_hat = self.decoder(z, seq_len)
        
        return x_hat, mean, log_var

    def _get_loss(self, batch, batch_idx):
        y, lengths = batch  # We do not need the x inputs, get only the labels y
        
        y_hat, mean, log_var = self.forward(y, lengths)

        recon_loss = self.loss_l1(y_hat, y, lengths)
        recon_loss = recon_loss.mean()

        kld = (- 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp()))

        ##### needs to be changed
        alpha = (((self.global_step+1) * self.batches_per_epoch) + batch_idx) / self.total_batches

        self.log("alpha", alpha)
        self.log("recon_loss", recon_loss)
        self.log("kld", kld)
        
        total_loss = recon_loss + (alpha * kld)

        return total_loss, recon_loss, alpha, kld

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def training_step(self, batch, batch_idx):
        total_loss, recon_loss, alpha, kld = self._get_loss(batch, batch_idx)

        self.log("train_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, recon_loss, alpha, kld = self._get_loss(batch, batch_idx)
        # print("total_loss: ", total_loss, "recon loss: ", recon_loss, "alpha: ", alpha, "kld: ", kld)
        self.log("val_loss", total_loss)

    def test_step(self, batch, batch_idx):
        total_loss, recon_loss, alpha, kld = self._get_loss(batch, batch_idx)
        self.log("test_loss", total_loss)

def train_vae(latent_dim, save_loc, train_loader, test_loader, num_epochs, bs, wandb_logger, hidden_dims, num_layers, bidirectional, lr):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=save_loc,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=bs,
        max_epochs=num_epochs,
        logger=wandb_logger,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(save_weights_only=True),
            L.pytorch.callbacks.LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    total_batches = len(train_loader) * num_epochs

    print("total batches:", total_batches)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = save_loc
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = VAE.load_from_checkpoint(pretrained_filename)
    else:
        model = VAE(latent_dim=latent_dim, num_batches=total_batches, num_epochs=num_epochs, hidden_dims=hidden_dims, num_layers=num_layers, bidirectional=bidirectional, bs=bs, lr=lr)
        # fit trainer
        trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = test_loader)
    # Test best model on test set
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result}
    return model, result