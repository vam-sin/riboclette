# libraries
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Dataset
from transformers import Trainer
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
import itertools
import argparse
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

def makePosEnc(seq_len, num_enc):
    '''
    make positional encodings
    '''
    pos_enc = np.zeros((seq_len, num_enc))
    for i in range(seq_len):
        for j in range(num_enc):
            if j % 2 == 0:
                pos_enc[i][j] = np.sin(i / 10000 ** (j / num_enc))
            else:
                pos_enc[i][j] = np.cos(i / 10000 ** ((j-1) / num_enc))

    return pos_enc

def RiboDataset(dataset_split, feature_folder, data_folder, model_type, transform, out_folder):
        
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

    for i in range(len(cbert_embeds)):
        print(i, len(cbert_embeds))

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
                if np.abs(ei[0][j] - ei[1][j]) == 1:
                    ea.append([0])
                else:
                    ea.append([1])

            ea = np.asarray(ea)
            ea = torch.from_numpy(ea).float()

            ew = torch.ones(ei.shape[1])

            for j in range(ei.shape[1]):
                if np.abs(ei[0][j] - ei[1][j]) != 1:
                    ew[j] = (np.abs(ei[0][j] - ei[1][j]) / len_Seq)

            # combine edge_attr and edge_weight into data.edge_attr 
            ea = torch.cat((ea, ew.unsqueeze(1)), dim=1)

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

            ea = np.asarray(ea)
            ea = torch.from_numpy(ea).float()

            ew = torch.ones(ei.shape[1])

            for j in range(ei.shape[1]):
                if np.abs(ei[0][j] - ei[1][j]) != 1:
                    ew[j] = (np.abs(ei[0][j] - ei[1][j]) / len_Seq)

            # combine edge_attr and edge_weight into data.edge_attr 
            ea = torch.cat((ea, ew.unsqueeze(1)), dim=1)

        # make positional encodings 
        pos_enc = makePosEnc(len_Seq, 32)

        # merge and then convert to torch tensor
        ft_vec = np.concatenate((cbert_embeds[i], pos_enc), axis=1)
        ft_vec = torch.from_numpy(ft_vec).float()
                    
        # ft_vec = torch.from_numpy(self.cbert_embeds[idx]).float()

        # output label
        y_i = norm_counts[i]
        y_i = y_i[1:-1].split(',')
        y_i = [float(el) for el in y_i]
        y_i = slidingWindowZeroToNan(y_i, window_size=30)
        y_i = [1 + el for el in y_i]
        y_i = np.asarray(y_i)
        y_i = np.log(y_i)
        y_i = torch.from_numpy(y_i).float()

        # make a data object
        # data = Data(edge_index = ei, x = ft_vec, y = y_i, edge_attr = ea)
        data = Data(edge_index = ei, x = ft_vec, y = y_i)

        # del ei, ft_vec, y_i, ea
        del ei, ft_vec, y_i

        # apply transform to data
        if transform:
            data = transform(data)

        torch.save(data, out_folder + dataset_split + '/sample_' + str(i) + '.pt')
        
if __name__ == '__main__':
    feature_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/'
    data_folder = '/nfs_home/nallapar/final/riboclette/riboclette/models/xlnet/data/sh/'
    random_walk_length = 32
    
    transforms = T.Compose([T.AddRandomWalkPE(walk_length=random_walk_length), T.VirtualNode()])

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='DirSeq+', help='condition to train on') # USeq, USeq+, DirSeq, DirSeq+
    args = parser.parse_args()
    model_type = args.model_type # USeq, USeq+, DirSeq, DirSeq+

    if model_type == 'USeq':
        out_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/USeq/' # USeq, USeqPlus, DirSeq, DirSeqPlus
    elif model_type == 'USeq+':
        out_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/USeqPlus/' # USeq, USeqPlus, DirSeq, DirSeqPlus
    elif model_type == 'DirSeq':
        out_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/DirSeq/' # USeq, USeqPlus, DirSeq, DirSeqPlus
    elif model_type == 'DirSeq+':
        out_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/DirSeqPlus/' # USeq, USeqPlus, DirSeq, DirSeqPlus

    print("Train Process")
    dat = RiboDataset('train', feature_folder, data_folder, model_type, transforms, out_folder=out_folder)
    print("Test Process")
    dat = RiboDataset('test', feature_folder, data_folder, model_type, transforms, out_folder=out_folder)