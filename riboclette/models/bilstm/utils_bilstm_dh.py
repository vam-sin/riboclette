# libraries
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer
from sklearn.model_selection import train_test_split
import itertools
import os
import lightning as L
from scipy import sparse
from torch.autograd import Variable

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
    longest = max(longest, current)
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

def slidingWindowZeroToNan(a, window_size=30):
    '''
    use a sliding window, if all the values in the window are 0, then replace them with nan
    '''
    a = np.asarray(a)
    for i in range(len(a) - window_size):
        if np.all(a[i:i+window_size] == 0.0):
            a[i:i+window_size] = np.nan

    return a

def coverageMod(a, window_size=30):
    '''
    returns the modified coverage function val in the sequence
    '''
    a = a[1:-1].split(', ')
    a = [float(k) for k in a]
    for i in range(len(a) - window_size):
        if np.all(a[i:i+window_size] == 0.0):
            a[i:i+window_size] = np.nan

    # num non zero, non nan
    num = 0
    den = 0
    for i in a:
        if i != 0.0 and not np.isnan(i):
            num += 1
        if not np.isnan(i):
            den += 1
    
    return num / den

def sequenceLength(a):
    '''
    returns the length of the sequence
    '''
    a = a[1:-1].split(', ')
    a = [float(k) for k in a]
    return len(a)

def uniqueGenes(df):
    # add sequence length column
    df['sequence_length'] = df['annotations'].apply(sequenceLength)
    # keep only longest transcript for each gene
    df = df.sort_values('sequence_length', ascending=False).drop_duplicates('gene').sort_index()
    # drop sequence length column
    df = df.drop(columns=['sequence_length'])

    assert len(df['gene'].unique()) == len(df['gene'])

    return df

def RiboDatasetGWSDepr(threshold: float = 0.6, longZerosThresh: int = 20, percNansThresh: float = 0.1):
    '''
    Dataset generation function
    '''
    # save the dataframes
    out_train_path = '../xlnet/data/dh/train_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '.csv'
    out_test_path = '../xlnet/data/dh/test_' + str(threshold) + '_NZ_' + str(longZerosThresh) + '_PercNan_' + str(percNansThresh) + '.csv'

    df_train = pd.read_csv(out_train_path)
    df_test = pd.read_csv(out_test_path)

    return df_train, df_test

class GWSDatasetFromPandas(Dataset):
    def __init__(self, df):
        self.df = df
        self.counts = list(self.df['annotations'])
        self.sequences = list(self.df['sequence'])
        self.condition_lists = list(self.df['condition'])
        self.condition_values = {'CTRL': 64, 'ILE': 65, 'LEU': 66, 'LEU_ILE': 67, 'LEU_ILE_VAL': 68, 'VAL': 69}

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

        y = slidingWindowZeroToNan(y)

        y = [1+i for i in y]
        y = np.log(y)

        # ctrl sequence 
        ctrl_y = self.df['ctrl_sequence'].iloc[idx]
        # convert string into list of floats
        ctrl_y = ctrl_y[1:-1].split(', ')
        ctrl_y = [float(i) for i in ctrl_y]

        ctrl_y = slidingWindowZeroToNan(ctrl_y)

        # no min max scaling
        ctrl_y = [1+i for i in ctrl_y]
        ctrl_y = np.log(ctrl_y)

        X = np.array(X)
        # multiply X with condition value times 64 + 1
        cond_token = self.condition_values[self.condition_lists[idx]]
        
        # prepend the condition token to X
        X = np.insert(X, 0, cond_token)

        y = np.array(y)

        X = torch.from_numpy(X).long()
        y = torch.from_numpy(y).float()
        ctrl_y = torch.from_numpy(ctrl_y).float()

        gene = self.df['gene'].iloc[idx]
        transcript = self.df['transcript'].iloc[idx]

        return X, y, ctrl_y, gene, transcript

class MaskedPearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, y_pred, y_true, mask, eps=1e-6):
        y_pred_mask = torch.masked_select(y_pred, mask)
        y_true_mask = torch.masked_select(y_true, mask)
        cos = nn.CosineSimilarity(dim=0, eps=eps)
        return 1 - cos(
            y_pred_mask - y_pred_mask.mean(),
            y_true_mask - y_true_mask.mean(),
        )

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        loss = nn.functional.l1_loss(y_pred_mask, y_true_mask, reduction="none")
        return torch.sqrt(loss.mean())

class MaskedPearsonCorr(nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, y_pred, y_true, mask, eps=1e-6):
        y_pred_mask = torch.masked_select(y_pred, mask)
        y_true_mask = torch.masked_select(y_true, mask)
        cos = nn.CosineSimilarity(dim=0, eps=eps)
        return cos(
            y_pred_mask - y_pred_mask.mean(),
            y_true_mask - y_true_mask.mean(),
        )

class MaskedCombinedFiveDH(nn.Module):
    def __init__(self):
        super().__init__()
        self.pearson = MaskedPearsonLoss()
        self.l1 = MaskedL1Loss()
        self.pearson_perf = MaskedPearsonCorr()
    
    def __call__(self, y_pred, labels, labels_ctrl, mask_full, mask_ctrl):
        # remove the first output cause that corresponds to the condition token
        y_pred_ctrl = y_pred[:, 0]
        y_pred_depr_diff = y_pred[:, 1]
        y_pred_full = torch.sum(y_pred, dim=1)

        labels_diff = labels - labels_ctrl

        # combine masks to make mask diff 
        mask_diff = mask_full & mask_ctrl

        loss_ctrl = self.pearson(y_pred_ctrl, labels_ctrl, mask_ctrl) + self.l1(y_pred_ctrl, labels_ctrl, mask_ctrl)
        loss_depr_diff = self.l1(y_pred_depr_diff, labels_diff, mask_diff)
        loss_full = self.pearson(y_pred_full, labels, mask_full) + self.l1(y_pred_full, labels, mask_full)

        perf = self.pearson_perf(y_pred_full, labels, mask_full)
        l1 = self.l1(y_pred_full, labels, mask_full)

        return loss_ctrl + loss_depr_diff + loss_full, perf, l1

class LSTM(L.LightningModule):
    def __init__(self, dropout_val, num_epochs, bs, lr):
        super().__init__()

        self.bilstm = nn.LSTM(128, 128, num_layers = 4, bidirectional=True)
        self.embedding = nn.Embedding(71, 128)
        self.linear = nn.Linear(256, 2) # double head
        
        self.relu = nn.ReLU()
        
        self.loss = MaskedCombinedFiveDH()
        self.perf = MaskedPearsonCorr()

        self.lr = lr
        self.bs = bs
        self.num_epochs = num_epochs
        self.perf_list = []
        self.mae_list = []
        self.out_tr = []

    def forward(self, x):
        # bilstm final layer
        h_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)

        # switch dims for lstm
        x = self.embedding(x)
        x = x.unsqueeze(dim=0)
        # print(x.shape)
        x = x.permute(1, 0, 2)

        x, (fin_h, fin_c) = self.bilstm(x, (h_0, c_0))

        # linear out
        x = self.linear(x)
        out = x.squeeze(dim=1)

        return out
    
    def _get_loss(self, batch):
        # get features and labels
        x, y, ctrl_y, g, t = batch

        y = y.squeeze(dim=0)

        # pass through model
        y_pred = self.forward(x)

        # remove the first dim of y_pred
        y_pred = y_pred[1:, :]

        # add dims

        # calculate masks
        lengths_full = torch.tensor([y.shape[0]]).to(y_pred)
        mask_full = torch.arange(y_pred.shape[0])[None, :].to(lengths_full) < lengths_full[:, None]
        mask_full = torch.logical_and(mask_full, torch.logical_not(torch.isnan(y)))

        # make mask for ctrl
        lengths_ctrl = torch.tensor([ctrl_y.shape[0]]).to(y_pred)
        mask_ctrl = torch.arange(y_pred.shape[0])[None, :].to(lengths_ctrl) < lengths_ctrl[:, None]
        mask_ctrl = torch.logical_and(mask_ctrl, torch.logical_not(torch.isnan(ctrl_y)))

        # squeeze mask
        mask_full = mask_full.squeeze(dim=0)
        mask_ctrl = mask_ctrl.squeeze(dim=0)

        loss, perf, mae = self.loss(y_pred, y, ctrl_y, mask_full, mask_ctrl)

        return loss, perf, mae

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        return [optimizer], [scheduler]
    
    def training_step(self, batch):

        loss, perf, mae = self._get_loss(batch)

        self.log('train/loss', loss, batch_size=self.bs)
        self.log('train/r', perf, batch_size=self.bs)

        return loss
    
    def validation_step(self, batch):
        loss, perf, mae = self._get_loss(batch)

        self.log('eval/loss', loss)
        self.log('eval/r', perf)

        return loss
    
    def test_step(self, batch):
        loss, perf, mae = self._get_loss(batch)

        self.log('test/loss', loss)
        self.log('test/r', perf)

        self.perf_list.append(perf.item())

        if len(self.perf_list) == 4252:
            # save perf list 
            df = pd.DataFrame(self.perf_list)
            df.to_csv('perf_list.csv')

        return loss
    
def trainLSTM(num_epochs, bs, lr, save_loc, wandb_logger, train_loader, test_loader, dropout_val):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=save_loc,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=bs,
        max_epochs=num_epochs,
        logger=wandb_logger,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(dirpath=save_loc,
                monitor='eval/loss',
                save_top_k=2),
            L.pytorch.callbacks.LearningRateMonitor("epoch"),
            L.pytorch.callbacks.EarlyStopping(monitor="eval/loss", patience=10),
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    # model = LSTM(dropout_val, num_epochs, bs, lr)
    # fit trainer
    # trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = test_loader)
    # # Test best model on test set
    # test_result = trainer.test(model, dataloaders=test_loader, verbose=False, ckpt_path="best")
    # result = {"test": test_result}

    # load model
    model = LSTM.load_from_checkpoint(save_loc+ '/epoch=2-step=53691.ckpt', dropout_val=dropout_val, num_epochs=num_epochs, bs=bs, lr=lr)

    # Test best model on test set
    test_result = trainer.test(model, dataloaders = test_loader, verbose=False)
    result = {"test": test_result}

    return model, result
    

        

