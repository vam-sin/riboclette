# libraries
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Dataset
from transformers import Trainer
from sklearn.model_selection import train_test_split
import itertools
import os
import lightning as L
from scipy import sparse
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.conv import SAGEConv, GATv2Conv
from torch_geometric.sampler import NeighborSampler
from torch_geometric.loader import NeighborLoader
from torch.autograd import Variable

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

def RiboDataset(dataset_split, feature_folder, data_folder, model_type, transform=None, edge_attr=False, sampler=False):
    # cbert features
    df_cbert = pd.read_pickle(feature_folder + 'LIVER_CodonBERT.pkl')

    # codon ss features
    df_codon_ss = pd.read_pickle(feature_folder + 'LIVER_VRNA_SS.pkl')

    # load datasets train and test
    if dataset_split == 'train':
        dataset_df = pd.read_csv(data_folder + 'train_OnlyLiver_Cov_0.3_NZ_20_PercNan_0.05.csv')
        transcripts = dataset_df['transcript'].tolist()

        df_cbert = df_cbert[df_cbert['transcript'].isin(transcripts)]
        cbert_embeds = df_cbert['cbert_embeds'].tolist()
        cbert_embeds_tr = df_cbert['transcript'].tolist()

        df_codon_ss = df_codon_ss[df_codon_ss['transcript'].isin(transcripts)]
        codon_ss = df_codon_ss['codon_RNA_SS'].tolist()

        # norm counts train and test
        norm_counts = [dataset_df[dataset_df['transcript'] == tr]['annotations'].values[0] for tr in cbert_embeds_tr]

    elif dataset_split == 'test':
        dataset_df = pd.read_csv(data_folder + 'test_OnlyLiver_Cov_0.3_NZ_20_PercNan_0.05.csv')
        transcripts = dataset_df['transcript'].tolist()

        df_cbert = df_cbert[df_cbert['transcript'].isin(transcripts)]
        cbert_embeds = df_cbert['cbert_embeds'].tolist()
        cbert_embeds_tr = df_cbert['transcript'].tolist()

        df_codon_ss = df_codon_ss[df_codon_ss['transcript'].isin(transcripts)]
        codon_ss = df_codon_ss['codon_RNA_SS'].tolist()

        # norm counts train and test
        norm_counts = [dataset_df[dataset_df['transcript'] == tr]['annotations'].values[0] for tr in cbert_embeds_tr]

    data_list = []
    for i in range(len(cbert_embeds)):
        full_adj_mat = np.asarray(codon_ss[i].todense())
        len_Seq = len(cbert_embeds[i])

        # make an undirected sequence graph
        if model_type == 'USeq':
            # make an undirected sequence graph
            adj_mat_useq = np.zeros((len_Seq, len_Seq))
            for j in range(len_Seq-1):
                adj_mat_useq[j][j+1] = 1
                adj_mat_useq[j+1][j] = 1
            
            # convert to sparse matrix
            adj_mat_useq = sparse.csr_matrix(adj_mat_useq) # USeq

            # convert to edge index and edge weight
            ei, ew = from_scipy_sparse_matrix(adj_mat_useq)

            # ea - edge attributes
            ea = []
            for j in range(len(ei[0])):
                ea.append([0])

        elif model_type == 'USeq+':
            # make an undirected sequence graph
            adj_mat_useq = np.zeros((len_Seq, len_Seq))
            for j in range(len_Seq-1):
                adj_mat_useq[j][j+1] = 1
                adj_mat_useq[j+1][j] = 1
            
            # subtract ribosome neighbourhood graph from sequence graph to get three d neighbours graph
            adj_mat_3d = full_adj_mat - adj_mat_useq # undirected 3d neighbours graph
            adj_mat_useqPlus = adj_mat_3d + adj_mat_useq

            # convert to sparse matrix
            adj_mat_useqPlus = sparse.csr_matrix(adj_mat_useqPlus) # USeq+

            # convert to edge index and edge weight
            ei, ew = from_scipy_sparse_matrix(adj_mat_useqPlus)

            # ea - edge attributes
            ea = []
            for j in range(len(ei[0])):
                if np.abs(ei[0][j] - ei[1][j]) == 1:
                    ea.append([0])
                else:
                    ea.append([1])

        elif model_type == 'DirSeq':
            # make a directed sequence graph
            adj_mat_dirseq = np.zeros((len_Seq, len_Seq))
            for j in range(1, len_Seq):
                adj_mat_dirseq[j][j-1] = 1 # A[i, j] = 1 denotes an edge from j to i

            # convert to sparse matrix
            adj_mat_dirseq = sparse.csr_matrix(adj_mat_dirseq) # DirSeq

            # convert to edge index and edge weight
            ei, ew = from_scipy_sparse_matrix(adj_mat_dirseq)

            # ea - edge attributes
            ea = []
            for j in range(len(ei[0])):
                ea.append([0])

        elif model_type == 'DirSeq+':
            # make an undirected sequence graph
            adj_mat_useq = np.zeros((len_Seq, len_Seq))
            for j in range(len_Seq-1):
                adj_mat_useq[j][j+1] = 1
                adj_mat_useq[j+1][j] = 1

            # make a directed sequence graph
            adj_mat_dirseq = np.zeros((len_Seq, len_Seq))
            for j in range(1, len_Seq):
                adj_mat_dirseq[j][j-1] = 1 # A[i, j] = 1 denotes an edge from j to i

            # subtract ribosome neighbourhood graph from sequence graph to get three d neighbours graph
            adj_mat_3d = full_adj_mat - adj_mat_useq # undirected 3d neighbours graph
            adj_mat_dirseqPlus = adj_mat_3d + adj_mat_dirseq # add the sequence as well, because just 3d neighbours might make it too sparse

            # convert to sparse matrix
            adj_mat_dirseqPlus = sparse.csr_matrix(adj_mat_dirseqPlus) # DirSeq+

            # convert to edge index and edge weight
            ei, ew = from_scipy_sparse_matrix(adj_mat_dirseqPlus)

            # ea - edge attributes
            ea = []
            for j in range(len(ei[0])):
                if np.abs(ei[0][j] - ei[1][j]) == 1:
                    ea.append([0])
                else:
                    ea.append([1])

        # merge and then convert to torch tensor
        ft_vec = torch.from_numpy(cbert_embeds[i]).float()

        # output label
        y_i = norm_counts[i]
        y_i = y_i[1:-1].split(',')
        y_i = [float(el) for el in y_i]
        y_i = slidingWindowZeroToNan(y_i, window_size=30)
        y_i = [1 + el for el in y_i]
        y_i = np.asarray(y_i)
        y_i = np.log(y_i)
        y_i = torch.from_numpy(y_i).float()

        # edge attr
        ea = np.asarray(ea)
        ea = torch.from_numpy(ea).float()

        # make a data object
        if edge_attr:
            data = Data(edge_index = ei, x = ft_vec, y = y_i, edge_attr = ea)
        else:
            data = Data(edge_index = ei, x = ft_vec, y = y_i)

        # apply transform to data
        if transform:
            data = transform(data)

        if sampler:
            loader = NeighborLoader(data, num_neighbors=[5] * 2, batch_size=1)
            sample = next(iter(loader))

            data_list.append(sample)
        else:
            data_list.append(data)

    return data_list
    
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

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        loss = nn.functional.l1_loss(y_pred_mask, y_true_mask, reduction="none")
        return torch.sqrt(loss.mean())
    
class MaskedPCCL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = MaskedL1Loss()
        self.pcc_loss = MaskedPearsonLoss()

    def __call__(self, y_pred, y_true, mask):
        '''
        loss is the sum of the l1 loss and the pearson correlation coefficient loss
        '''

        l1 = self.l1_loss(y_pred, y_true, mask)
        pcc = self.pcc_loss(y_pred, y_true, mask)

        return l1 + pcc, pcc

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, alpha, model_type, algo):
        super().__init__()

        if algo == 'SAGE':
            self.conv_in = SAGEConv(in_channels, out_channels)
            self.conv_out = SAGEConv(in_channels, out_channels)
        elif algo == 'GAT':
            self.conv_in = GATConv(in_channels, out_channels, heads = 8)
            self.conv_out = GATConv(in_channels, out_channels, heads = 8)

        self.alpha = alpha
        self.model_type = model_type

    def forward(self, x, ei):
        if self.model_type == 'USeq' or self.model_type == 'USeq+':
            x_in = self.conv_in(x, ei)

            return x_in

        elif self.model_type == 'DirSeq' or self.model_type == 'DirSeq+':
            x_in = self.conv_in(x, ei)
            x_out = self.conv_out(x, ei.flip(dims=(0,)))

            return x_in + x_out

class GCN(L.LightningModule):
    def __init__(self, gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, alpha, model_type, algo):
        super().__init__()
    
        self.gcn_layers = gcn_layers

        self.module_list = nn.ModuleList()
        self.graph_norm_list = nn.ModuleList()

        self.module_list.append(ConvModule(num_inp_ft, gcn_layers[0], alpha, model_type, algo)) 
        self.graph_norm_list.append(GraphNorm(gcn_layers[0]))
        
        for i in range(len(gcn_layers)-1):
            self.module_list.append(ConvModule(gcn_layers[i], gcn_layers[i+1], alpha, model_type, algo))
            self.graph_norm_list.append(GraphNorm(gcn_layers[i+1]))

        self.dropout = nn.Dropout(dropout_val)

        self.bilstm = nn.LSTM(np.sum(gcn_layers), 256, num_layers = 4, bidirectional=True)

        self.linear = nn.Linear(512, 1)
        # self.linear = nn.Linear(np.sum(gcn_layers), 1)
        
        self.relu = nn.ReLU()
        
        self.loss = MaskedPCCL1Loss()
        self.perf = MaskedPearsonCorr()

        self.lr = lr
        self.bs = bs
        self.num_epochs = num_epochs

        self.model_type = model_type
        self.algo = algo

    def forward(self, data):
        x, ei = data.x, data.edge_index

        outputs = []

        for i in range(len(self.gcn_layers)):
            x = self.module_list[i](x, ei)
            
            # only for GAT
            if self.algo == 'GAT':
                x = x.reshape(x.shape[0], self.gcn_layers[i], 8)
                # mean over final dimension
                x = torch.mean(x, dim=2)
            
            x = self.graph_norm_list[i](x)
            x = self.relu(x)
            x = self.dropout(x)

            outputs.append(x)

        out = torch.cat(outputs, dim=1)

        # bilstm final layer
        h_0 = Variable(torch.zeros(8, 1, 256).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(8, 1, 256).cuda()) # (1, bs, hidden)

        out = out.unsqueeze(1)
        out, (fin_h, fin_c) = self.bilstm(out, (h_0, c_0))

        # linear out
        out = self.linear(out)
        out = out.squeeze(dim=1)
        
        # extra for lstm
        out = out.squeeze(dim=1)

        return out
    
    def _get_loss(self, batch):
        # get features and labels
        batch = batch[0]
        y = batch.y

        # pass through model
        y_pred = self.forward(batch)
        
        # # remove virtual node from y_pred (other way) - this is better
        # y_pred = y_pred[:-1]

        # calculate loss
        lengths = torch.tensor([y.shape[0]]).to(y_pred)
        mask = torch.arange(y_pred.shape[0])[None, :].to(lengths) < lengths[:, None]
        mask = torch.logical_and(mask, torch.logical_not(torch.isnan(y)))

        # squeeze mask
        mask = mask.squeeze(dim=0)

        loss, cos = self.loss(y_pred, y, mask)

        perf = 1 - cos

        return loss, perf

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        return [optimizer], [scheduler]
    
    def training_step(self, batch):
        loss, perf = self._get_loss(batch)

        self.log('train_loss', loss, batch_size=self.bs)
        self.log('train_perf', perf, batch_size=self.bs)

        return loss
    
    def validation_step(self, batch):
        loss, perf = self._get_loss(batch)

        self.log('val_perf', perf)
        self.log('val_loss', loss)

        return loss
    
    def test_step(self, batch):
        loss, perf = self._get_loss(batch)

        self.log('test_loss', loss)
        self.log('test_perf', perf)

        return loss
    
def trainGCN(gcn_layers, num_epochs, bs, lr, save_loc, wandb_logger, train_loader, test_loader, dropout_val, num_inp_ft, alpha, model_type, algo):
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
                monitor='val_loss',
                save_top_k=2),
            L.pytorch.callbacks.LearningRateMonitor("epoch"),
            L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=20),
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    total_batches = len(train_loader) * num_epochs

    # Check whether pretrained model exists. If yes, load it and skip training
    model = GCN(gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, alpha, model_type, algo)
    # fit trainer
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = test_loader)
    # Test best model on test set
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False, ckpt_path="best")
    result = {"test": test_result}
    return model, result
    

        

