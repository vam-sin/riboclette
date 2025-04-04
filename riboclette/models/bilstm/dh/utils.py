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
from pyhere import here

id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

def slidingWindowZeroToNan(a, window_size=30):
    '''
    use a sliding window, if all the values in the window are 0, then replace them with nan
    '''
    a = np.asarray(a)
    for i in range(len(a) - window_size):
        if np.all(a[i:i+window_size] == 0.0):
            a[i:i+window_size] = np.nan

    return a

def RiboDatasetGWSDepr():
    '''
    Dataset generation function
    '''
    train_path = here('data', 'orig', 'train.csv')
    val_path = here('data', 'orig', 'val.csv')
    test_path = here('data', 'orig', 'test.csv')

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_val, df_test

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
        condition = self.df['condition'].iloc[idx]

        return X, y, ctrl_y, gene, transcript, condition

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
    
class MaskedCombinedFourDH(nn.Module):
    def __init__(self):
        super().__init__()
        self.pearson = MaskedPearsonLoss()
        self.l1 = MaskedL1Loss()
        self.pearson_perf = MaskedPearsonCorr()
    
    def __call__(self, y_pred, labels, labels_ctrl, mask_full, mask_ctrl, condition_):
        # remove the first output cause that corresponds to the condition token
        y_pred_ctrl = y_pred[:, 0]
        y_pred_depr_diff = y_pred[:, 1]
        y_pred_full = torch.sum(y_pred, dim=1)

        labels_diff = labels - labels_ctrl

        # combine masks to make mask diff 
        mask_diff = mask_full & mask_ctrl

        loss_ctrl = self.pearson(y_pred_ctrl, labels_ctrl, mask_ctrl) 
        if condition_ != 64:
            loss_depr_diff = self.pearson(y_pred_depr_diff, labels_diff, mask_diff)
        loss_full = self.pearson(y_pred_full, labels, mask_full) + self.l1(y_pred_full, labels, mask_full)

        perf = self.pearson_perf(y_pred_full, labels, mask_full)
        l1 = self.l1(y_pred_full, labels, mask_full)

        if condition_ != 64:
            return loss_ctrl + loss_depr_diff + loss_full, perf, l1
        else:
            return loss_ctrl + loss_full, perf, l1

class LSTM(L.LightningModule):
    def __init__(self, dropout_val, num_epochs, bs, lr, num_layers, num_nodes):
        super().__init__()

        self.bilstm = nn.LSTM(num_nodes, num_nodes, num_layers = num_layers, bidirectional=True)
        self.embedding = nn.Embedding(71, num_nodes)
        self.linear = nn.Linear(num_nodes * 2, 2) # double head
        
        self.relu = nn.ReLU()
        
        self.loss = MaskedCombinedFourDH()
        self.perf = MaskedPearsonCorr()

        self.lr = lr
        self.bs = bs
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.num_epochs = num_epochs
        self.perf_list = []
        self.mae_list = []
        self.conds_list = []
        self.out_tr = []

    def forward(self, x):
        # bilstm final layer
        h_0 = Variable(torch.zeros(self.num_layers*2, 1, self.num_nodes).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(self.num_layers*2, 1, self.num_nodes).cuda()) # (1, bs, hidden)

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
        x, y, ctrl_y, g, t, c = batch

        y = y.squeeze(dim=0)

        # pass through model
        y_pred = self.forward(x)

        # remove the first dim of y_pred
        y_pred = y_pred[1:, :]

        # condition
        condition_ = x[0].item()

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

        loss, perf, mae = self.loss(y_pred, y, ctrl_y, mask_full, mask_ctrl, condition_)

        return loss, perf, mae, c

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        return [optimizer], [scheduler]
    
    def training_step(self, batch):

        loss, perf, mae, c = self._get_loss(batch)

        self.log('train/loss', loss, batch_size=self.bs)
        self.log('train/r', perf, batch_size=self.bs)

        return loss
    
    def validation_step(self, batch):
        loss, perf, mae, c = self._get_loss(batch)

        self.log('eval/loss', loss)
        self.log('eval/r', perf)

        return loss
    
    def test_step(self, batch):
        loss, perf, mae, c = self._get_loss(batch)

        self.log('test/loss', loss)
        self.log('test/r', perf)

        self.perf_list.append(perf.item())
        self.conds_list.append(c)

        return loss
    
def trainLSTM(num_epochs, bs, lr, save_loc, train_loader, test_loader, val_loader, dropout_val, num_layers, num_nodes):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=save_loc,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=bs,
        max_epochs=num_epochs,
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
    model = LSTM(dropout_val, num_epochs, bs, lr, num_layers, num_nodes)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total LSTM parameters: ", pytorch_total_params)
    # fit trainer
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)
    # Test best model on test set
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result}

    return model, result
    

        

