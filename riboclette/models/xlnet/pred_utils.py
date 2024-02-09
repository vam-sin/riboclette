# %%
# libraries
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import math
import os
from scipy import stats
from scipy.stats import pearsonr, spearmanr 
from torchmetrics.functional import pearson_corrcoef
import itertools
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.interpolate import make_interp_spline 
from captum.attr import IntegratedGradients, LayerIntegratedGradients, LayerGradientXActivation
import matplotlib.pyplot as plt
sns.set_style("white")

# %%
# global variables defined for all the function to use
# one-hot encoding for the conditions
condition_values = {'CTRL': 64, 'ILE': 65, 'LEU': 66, 'LEU_ILE': 67, 'LEU_ILE_VAL': 68, 'VAL': 69}
inverse_condition_values = {64: 'CTRL', 65: 'ILE', 66: 'LEU', 67: 'LEU_ILE', 68: 'LEU_ILE_VAL', 69: 'VAL'}

# one-hot encoding for the codons
id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

# %%
# metric functions
def pearson_mask(pred, label):
    '''
    inputs: model prediction, true label
    outputs: pearson correlation coefficient
    '''
    # take the prediction and label
    full_pred_tensor = torch.tensor(pred)
    label_tensor = torch.tensor(label)

    # make mask tensor
    # remove the end token from the mask
    mask = label_tensor != -100.0
    mask = torch.tensor(mask)

    # remove the nans from the mask
    mask = torch.logical_and(mask, torch.logical_not(torch.isnan(label_tensor)))
    # double mask
    mask = torch.logical_and(mask, torch.logical_not(torch.isnan(full_pred_tensor)))

    # set model prediction to same length as label tensor
    full_pred_tensor = full_pred_tensor[:len(mask)]

    assert full_pred_tensor.shape == label_tensor.shape
    assert label_tensor.shape == mask.shape

    # select the elements from the tensors that are not nan
    mp, mt = torch.masked_select(full_pred_tensor, mask), torch.masked_select(label_tensor, mask)

    # calculate pearson correlation coefficient
    temp_pearson = pearson_corrcoef(mp, mt)

    # get float value from tensor
    temp_pearson = temp_pearson.item()

    return temp_pearson

def mae_mask(pred, label):
    '''
    inputs: model prediction, true label
    outputs: mean absolute error
    '''
    # take the prediction and label
    full_pred_tensor = torch.tensor(pred)
    label_tensor = torch.tensor(label)

    # make mask tensor
    # remove the end token from the mask
    mask = label_tensor != -100.0
    mask = torch.tensor(mask)

    # remove the nans from the mask
    mask = torch.logical_and(mask, torch.logical_not(torch.isnan(label_tensor)))
    # double mask
    mask = torch.logical_and(mask, torch.logical_not(torch.isnan(full_pred_tensor)))

    # set model prediction to same length as label tensor
    full_pred_tensor = full_pred_tensor[:len(mask)]

    assert full_pred_tensor.shape == label_tensor.shape
    assert label_tensor.shape == mask.shape

    # select the elements from the tensors that are not nan
    mp, mt = torch.masked_select(full_pred_tensor, mask), torch.masked_select(label_tensor, mask)

    # calculate mean absolute error
    temp_mae = torch.mean(torch.abs(mp - mt))

    # get float value from tensor
    temp_mae = temp_mae.item()

    return temp_mae

def mape_mask(pred, label):
    '''
    inputs: model prediction, true label
    outputs: mean absolute error
    '''
    # take the prediction and label
    full_pred_tensor = torch.tensor(pred)
    label_tensor = torch.tensor(label)

    # make mask tensor
    # remove the end token from the mask
    mask = label_tensor != -100.0
    mask = torch.tensor(mask)

    # remove the nans from the mask
    mask = torch.logical_and(mask, torch.logical_not(torch.isnan(label_tensor)))
    # double mask
    mask = torch.logical_and(mask, torch.logical_not(torch.isnan(full_pred_tensor)))

    # set model prediction to same length as label tensor
    full_pred_tensor = full_pred_tensor[:len(mask)]

    assert full_pred_tensor.shape == label_tensor.shape
    assert label_tensor.shape == mask.shape

    # select the elements from the tensors that are not nan
    mp, mt = torch.masked_select(full_pred_tensor, mask), torch.masked_select(label_tensor, mask)

    # calculate mean absolute percentage error
    temp_mape = torch.mean(torch.abs((mp - mt) / mt))

    # get float value from tensor
    temp_mape = temp_mape.item()

    return temp_mape

def f1_score_masked(pred, label):
    '''
    inputs: model prediction, true label
    outputs: f1 score
    '''
    # take the prediction and label
    full_pred_tensor = torch.tensor(pred)
    label_tensor = torch.tensor(label)

    # make mask tensor
    # remove the end token from the mask
    mask = label_tensor != -100.0
    mask = torch.tensor(mask)

    # remove the nans from the mask
    mask = torch.logical_and(mask, torch.logical_not(torch.isnan(label_tensor)))

    # set model prediction to same length as label tensor
    full_pred_tensor = full_pred_tensor[:len(mask)]

    assert full_pred_tensor.shape == label_tensor.shape
    assert label_tensor.shape == mask.shape

    # select the elements from the tensors that are not nan
    mp, mt = torch.masked_select(full_pred_tensor, mask), torch.masked_select(label_tensor, mask)

    # calculate f1 score
    temp_f1 = f1_score(mp, mt, average='macro')

    # get float value from tensor
    temp_f1 = temp_f1.item()

    return temp_f1

def prec_score_masked(pred, label):
    '''
    inputs: model prediction, true label
    outputs: precision score
    ''' 
    # take the prediction and label
    full_pred_tensor = torch.tensor(pred)
    label_tensor = torch.tensor(label)

    # make mask tensor
    # remove the end token from the mask
    mask = label_tensor != -100.0
    mask = torch.tensor(mask)

    # remove the nans from the mask
    mask = torch.logical_and(mask, torch.logical_not(torch.isnan(label_tensor)))

    # set model prediction to same length as label tensor
    full_pred_tensor = full_pred_tensor[:len(mask)]

    assert full_pred_tensor.shape == label_tensor.shape
    assert label_tensor.shape == mask.shape

    # select the elements from the tensors that are not nan
    mp, mt = torch.masked_select(full_pred_tensor, mask), torch.masked_select(label_tensor, mask)

    # calculate precision score
    temp_prec = precision_score(mp, mt, average='macro')

    # get float value from tensor
    temp_prec = temp_prec.item()

    return temp_prec 

def recall_score_masked(pred, label):
    '''
    inputs: model prediction, true label
    outputs: recall score
    '''
    # take the prediction and label
    full_pred_tensor = torch.tensor(pred)
    label_tensor = torch.tensor(label)

    # make mask tensor
    # remove the end token from the mask
    mask = label_tensor != -100.0
    mask = torch.tensor(mask)

    # remove the nans from the mask
    mask = torch.logical_and(mask, torch.logical_not(torch.isnan(label_tensor)))

    # set model prediction to same length as label tensor
    full_pred_tensor = full_pred_tensor[:len(mask)]

    assert full_pred_tensor.shape == label_tensor.shape
    assert label_tensor.shape == mask.shape

    # select the elements from the tensors that are not nan
    mp, mt = torch.masked_select(full_pred_tensor, mask), torch.masked_select(label_tensor, mask)

    # calculate recall score
    temp_rec = recall_score(mp, mt, average='macro')

    # get float value from tensor
    temp_rec = temp_rec.item()

    return temp_rec

# %%
class model_finalexp(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model 
    
    def forward(self, x, index_val):
        # input dict
        out_batch = {}

        out_batch["input_ids"] = x

        # move to device
        for k, v in out_batch.items():
            out_batch[k] = v.to(device)

        out_batch["input_ids"] = torch.tensor(out_batch["input_ids"]).to(device).to(torch.int32)

        pred = self.model(out_batch["input_ids"])

        # add the values in the last dims
        pred_fin = torch.sum(pred["logits"], dim=2)

        # squeeze
        pred_fin = pred_fin.squeeze(0)

        out = pred_fin[index_val].unsqueeze(0)

        return out 
    
class model_finalexpLIG(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model 
    
    def forward(self, x, index_val):
        # input dict
        out_batch = {}

        out_batch["input_ids"] = x.unsqueeze(0)
        for k, v in out_batch.items():
            out_batch[k] = v.to(device)

        out_batch["input_ids"] = torch.tensor(out_batch["input_ids"]).to(device).to(torch.int32)

        pred = self.model(out_batch["input_ids"])

        pred_fin = torch.sum(pred["logits"], dim=2)

        pred_fin = pred_fin.squeeze(0)

        out = pred_fin[index_val].unsqueeze(0)

        return out 

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def attention_output(model, x, y, ctrl_y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    list_attn_matrices = []
    max_len = 0
    lens_list = []
    with torch.no_grad():
        lengths = torch.tensor([len(x)])

        x = torch.tensor(x).unsqueeze(0)
        y = torch.tensor(y).unsqueeze(0)
        ctrl_y = torch.tensor(ctrl_y).unsqueeze(0)
        
        x = pad_sequence(x, batch_first=True, padding_value=70) 
        y = pad_sequence(y, batch_first=True, padding_value=-1)
        ctrl_y = pad_sequence(ctrl_y, batch_first=True, padding_value=-1)

        out_batch = {}

        out_batch["input_ids"] = x
        out_batch["labels"] = y
        out_batch["lengths"] = lengths
        out_batch["labels_ctrl"] = ctrl_y

        # send batch to device
        for k, v in out_batch.items():
            out_batch[k] = v.to(device)

        out = model(out_batch["input_ids"], output_attentions = True, return_dict = True)
        attn_vec1 = out.attentions[0].cpu().detach().numpy()

        attn_vec_full = attn_vec1 # only first layer because this is the only one that looks at the input

        # remove dim 0
        attn_vec_full = np.squeeze(attn_vec_full, axis=0)

        # average across heads
        attn_vec_full = np.mean(attn_vec_full, axis=0)

    # make dataframe position_to_Asite and the attention weight for that position
    pos_A_site_df = []
    attn_weight_df = []
    for i in range(len(attn_vec_full)):
        for j in range(len(attn_vec_full)):
            pos_A_site_df.append(j-i)
            attn_weight_df.append(np.abs(attn_vec_full[i][j]))

    df_attn_weights = pd.DataFrame({'Relative Distance from A Site': pos_A_site_df, 'Attention Weight': attn_weight_df})

    return df_attn_weights

def layergradactivation_output(model, x):
    model_fin = model_finalexp(model)
        
    lig = LayerGradientXActivation(model_fin, model_fin.model.transformer.word_embedding)

    # set torch graph to allow unused tensors
    with torch.autograd.set_detect_anomaly(True):
        x = torch.tensor(x).unsqueeze(0)
        
        x = pad_sequence(x, batch_first=True, padding_value=70) 

        out_batch = {}

        out_batch["input_ids"] = x
        
        out_batch["input_ids"] = torch.tensor(out_batch["input_ids"]).to(device).to(torch.int32)
        
        # make len(x[0]) x len(x[0]) matrix
        len_sample = len(x[0])
        attributions_sample = np.zeros((len_sample, len_sample))

        for j in range(len_sample):
            index_val = j

            index_val = torch.tensor(index_val).to(device)

            attributions = lig.attribute((out_batch["input_ids"]), additional_forward_args=index_val)
            attributions = attributions.squeeze(1)
            attributions = torch.sum(attributions, dim=1)
            attributions = attributions / torch.norm(attributions)
            attributions = attributions.detach().cpu().numpy()
            attributions_sample[j] = attributions
        
        attributions_sample = np.array(attributions_sample)

    # make dataframe position_to_Asite and the attention weight for that position
    pos_A_site_df = []
    attr_weight_df = []
    for i in range(len(attributions_sample)):
        for j in range(len(attributions_sample)):
            pos_A_site_df.append(j-i)
            attr_weight_df.append(np.abs(attributions_sample[i][j]))

    df_attr_weights = pd.DataFrame({'Relative Distance from A Site': pos_A_site_df, 'GradxAct Weight': attr_weight_df})

    return df_attr_weights

def integratedgrad_output(model, x):
    model_fin = model_finalexpLIG(model)
        
    lig = LayerIntegratedGradients(model_fin, model_fin.model.transformer.word_embedding)

    # set torch graph to allow unused tensors
    with torch.autograd.set_detect_anomaly(True):
        x = torch.tensor(x)

        out_batch = {}

        out_batch["input_ids"] = x
        
        out_batch["input_ids"] = torch.tensor(out_batch["input_ids"]).to(device).to(torch.int32)

        baseline_inp = torch.ones(out_batch["input_ids"].shape) * 70
        baseline_inp = baseline_inp.to(device).to(torch.int32)
        
        # make len(x[0]) x len(x[0]) matrix
        len_sample = len(x)
        attributions_sample = np.zeros((len_sample, len_sample))

        for j in range(len_sample):
            index_val = j

            index_val = torch.tensor(index_val).to(device)

            attributions, approximation_error = lig.attribute((out_batch["input_ids"]), baselines=baseline_inp, 
                                                    method = 'gausslegendre', return_convergence_delta = True, additional_forward_args=index_val, n_steps=20, internal_batch_size=2048)

            
            attributions = attributions.squeeze(1)
            attributions = torch.sum(attributions, dim=1)
            attributions = attributions / torch.norm(attributions)
            attributions = attributions.detach().cpu().numpy()
            attributions_sample[j] = attributions
        
        attributions_sample = np.array(attributions_sample)

    # make dataframe position_to_Asite and the attention weight for that position
    pos_A_site_df = []
    attr_weight_df = []
    for i in range(len(attributions_sample)):
        for j in range(len(attributions_sample)):
            pos_A_site_df.append(j-i)
            attr_weight_df.append(np.abs(attributions_sample[i][j]))

    df_attr_weights = pd.DataFrame({'Relative Distance from A Site': pos_A_site_df, 'LIG Weight': attr_weight_df})

    return df_attr_weights

def interpretability_plot(model, x, y, ctrl_y, pred_y, save_path, pred_pcc, transcript, gene, condition_str):
    y = np.array(y)
    pred_y = np.array(pred_y)
    codon_position = np.array([i for i in range(len(y))])
    ylabel_df = pd.DataFrame({'Codon Position': codon_position, 'Ribosome Density': y})
    pred_df = pd.DataFrame({'Codon Position': codon_position, 'Ribosome Density': pred_y})
    attn_df = attention_output(model, x, y, ctrl_y)
    print('attention done')
    gradxact_df = layergradactivation_output(model, x)
    print('gradxact done')
    intgrad_df = integratedgrad_output(model, x)
    print('intgrad done')

    print('making plots')

    # 5 line plots: attention, layergradactivation, integratedgrad, label y, predicted y
    # make the plots
    fig, axs = plt.subplots(5, 1, figsize=(30, 10))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Interpretability Plots for the Model', fontsize=20)
    # add transcript, gene, condition, and pred_pcc info to the title
    fig.suptitle('Transcript: ' + transcript + ' Gene: ' + gene + ' Condition: ' + condition_str + '\n Predicted PCC: ' + str(pred_pcc), fontsize=20)

    # label y with array y 
    sns.lineplot(data=ylabel_df, x='Codon Position', y='Ribosome Density', ax=axs[0], color='#f1c40f', label='Label')

    # predicted y with array pred_y
    sns.lineplot(data=pred_df, x='Codon Position', y='Ribosome Density', ax=axs[1], color='#3498db', label='Predicted')

    # attention plot
    sns.lineplot(data=attn_df, x='Relative Distance from A Site', y='Attention Weight', ax=axs[2], color='#e74c3c', label='Attention')

    # layergradactivation plot
    sns.lineplot(data=gradxact_df, x='Relative Distance from A Site', y='GradxAct Weight', ax=axs[3], color='#2ecc71', label='GradxAct')

    # integratedgrad plot
    sns.lineplot(data=intgrad_df, x='Relative Distance from A Site', y='LIG Weight', ax=axs[4], color='#30336b', label='LIG')

    # save the plot
    plt.savefig(save_path)
    



# %%
def make_plot(full_preds_sample, depr_diffs_sample, ctrl_preds_sample, labels_sample, labels_ctrl_sample, labels_depr_diff_sample, out_loc_sample, transcript_sample, gene_sample, inverse_condition_values_sample):
    pearson_corr_full = pearson_mask(full_preds_sample, labels_sample)
    mae_full = mae_mask(full_preds_sample, labels_sample)
    pearson_corr_ctrl = pearson_mask(ctrl_preds_sample, labels_ctrl_sample)
    mae_ctrl = mae_mask(ctrl_preds_sample, labels_ctrl_sample)

    mae_dd = mae_mask(depr_diffs_sample, labels_depr_diff_sample)

    plot_title = "Transcript: " + str(transcript_sample) + " Gene: " + str(gene_sample) + " Condition: " + str(inverse_condition_values_sample) + "\n\nPearson Correlation " + str(inverse_condition_values_sample) + ": " + str(pearson_corr_full) + " || MAE " + str(inverse_condition_values_sample) + ":" + str(mae_full) + "\n\nPearson Correlation CTRL: " + str(pearson_corr_ctrl) + " || MAE CTRL: " + str(mae_ctrl) + "\n\n" + "MAE Depr Diffs: " + str(mae_dd) + "\n\n"
    
    print(plot_title)    

    # set min and max y values for the sub plot
    # control
    min_y_c = min(min(labels_ctrl_sample), min(ctrl_preds_sample)) - 0.1
    max_y_c = max(max(labels_ctrl_sample), max(ctrl_preds_sample)) + 0.1

    # deprivation difference
    min_y_diff = min(min(depr_diffs_sample), min(labels_depr_diff_sample)) - 0.1
    max_y_diff = max(max(depr_diffs_sample), max(labels_depr_diff_sample)) + 0.1

    # full prediction
    min_y_f = min(min(labels_sample), min(full_preds_sample)) - 0.1
    max_y_f = max(max(labels_sample), max(full_preds_sample)) + 0.1

    # subplots for ctrl, deprivation difference, and full prediction (with labels)
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))
    # add title 
    fig.suptitle(plot_title, fontsize=16)
    # add space after title
    fig.tight_layout(pad=10.0)

    # FIRST SUBPLOT: CTRL
    # make it bar plots
    x = np.arange(len(ctrl_preds_sample))
    # axs[0, 0].bar(height = ctrl_preds_sample, x = x, color='#00A757', label='CTRL Prediction')
    # axs[0, 1].bar(height = labels_ctrl_sample, x = x, color='#82BA4F', label='CTRL Label')

    axs[0, 0].plot(ctrl_preds_sample, color='#00A757', label='CTRL Prediction')
    axs[0, 1].plot(labels_ctrl_sample, color='#82BA4F', label='CTRL Label')

    # make a vector marking the nans with 0, and the rest of the values with nan
    # make a vector of nans the same size as ctrl_preds_sample
    labels_ctrl_nans = np.empty(len(labels_ctrl_sample))
    labels_ctrl_nans[:] = np.nan
    for k in range(len(labels_ctrl_sample)):
        if np.isnan(labels_ctrl_sample[k]):
            labels_ctrl_nans[k] = 0
            try:
                labels_ctrl_nans[k+1] = 0
                labels_ctrl_nans[k-1] = 0
            except:
                pass

    label_ctrl_zeros = np.empty(len(labels_ctrl_sample))
    for k in range(len(labels_ctrl_sample)):
        if labels_ctrl_sample[k]:
            label_ctrl_zeros[k] = np.nan
        else:
            label_ctrl_zeros[k] = 0

    # plot the nans
    axs[0, 1].plot(labels_ctrl_nans, color='black')
    axs[0, 1].plot(label_ctrl_zeros, color='#82BA4F')

    # set y limits
    axs[0, 0].set_ylim([min_y_c, max_y_c])
    axs[0, 1].set_ylim([min_y_c, max_y_c])

    # SECOND SUBPLOT: DEPRIVATION DIFFERENCE
    x = np.arange(len(depr_diffs_sample))
    # axs[1, 0].bar(height = depr_diffs_sample, x = x, color='#C82E6B', label='Deprivation Difference Prediction')
    # axs[1, 1].bar(height = labels_depr_diff_sample, x = x, color='#D4668F', label='Deprivation Difference Label')

    axs[1, 0].plot(depr_diffs_sample, color='#C82E6B', label='Deprivation Difference Prediction')
    axs[1, 1].plot(labels_depr_diff_sample, color='#D4668F', label='Deprivation Difference Label')

    # make a vector marking the nans in labels_depr_diff_sample
    labels_depr_diff_nans = np.empty(len(labels_depr_diff_sample))
    labels_depr_diff_nans[:] = np.nan
    for k in range(len(labels_depr_diff_sample)):
        if np.isnan(labels_depr_diff_sample[k]):
            labels_depr_diff_nans[k] = 0
            try:
                labels_depr_diff_nans[k+1] = 0
                labels_depr_diff_nans[k-1] = 0
            except:
                pass

    labels_depr_diff_zeros = np.empty(len(labels_depr_diff_sample))
    for k in range(len(labels_depr_diff_sample)):
        if labels_depr_diff_sample[k]:
            labels_depr_diff_zeros[k] = np.nan
        else:
            labels_depr_diff_zeros[k] = 0

    # plot the nans
    axs[1, 1].plot(labels_depr_diff_nans, color='black')
    axs[1, 1].plot(labels_depr_diff_zeros, color='#D4668F')

    # set y limits
    axs[1, 0].set_ylim([min_y_diff, max_y_diff])
    axs[1, 1].set_ylim([min_y_diff, max_y_diff])

    # THIRD SUBPLOT: DEPRIVATION FULL PREDICTION
    x = np.arange(len(full_preds_sample))
    # axs[2, 0].bar(height = full_preds_sample, x = x, color='#65BADA', label= inverse_condition_values_sample + ' Prediction')
    # axs[2, 1].bar(height = labels_sample, x = x, color='#87D0E2', label=inverse_condition_values_sample + ' Label')

    axs[2, 0].plot(full_preds_sample, color='#65BADA', label= inverse_condition_values_sample + ' Prediction')
    axs[2, 1].plot(labels_sample, color='#87D0E2', label=inverse_condition_values_sample + ' Label')

    # make a vector marking the nans in labels_sample
    labels_full_nans = np.empty(len(labels_sample))
    labels_full_nans[:] = np.nan
    for k in range(len(labels_sample)):
        if np.isnan(labels_sample[k]):
            labels_full_nans[k] = 0
            try:
                labels_full_nans[k+1] = 0
                labels_full_nans[k-1] = 0
            except:
                pass

    labels_full_zeros = np.empty(len(labels_sample))
    for k in range(len(labels_sample)):
        if labels_sample[k]:
            labels_full_zeros[k] = np.nan
        else:
            labels_full_zeros[k] = 0

    # plot the nans
    axs[2, 1].plot(labels_full_nans, color='black')
    axs[2, 1].plot(labels_full_zeros, color='#87D0E2')

    # set y limits
    axs[2, 0].set_ylim([min_y_f, max_y_f])
    axs[2, 1].set_ylim([min_y_f, max_y_f])

    # set x and y labels for all the subplots 
    for k in range(3):
        for j in range(2):
            axs[k, j].set_xlabel('Codon Position', fontsize=16)
            axs[k, j].set_ylabel('Ribosome Read Counts', fontsize=16)
            axs[k, j].legend(fontsize=16, loc="upper right")

    fig.tight_layout()

    plt.savefig(out_loc_sample)

    # display the plot
    plt.show()
    plt.clf()

def checkArrayEquality(arr1, arr2):
    '''
    inputs: two arrays
    outputs: True if the arrays are equal, False otherwise
    '''
    if len(arr1) != len(arr2):
        return False
    
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return False
    
    return True

def analyse_dh_outputs(preds, labels, inputs, output_loc, test_data_path):
    '''
    inputs: model predictions, true labels, inputs, output folder path
    outputs: 
    1. prints the condition wise mean pearson correlation coefficient
    2. generates plots for the two heads (ctrl, and deprivation difference), full prediction, and the two labels (ctrl, and deprivation) for the 10 best and 10 worst performing transcripts
    '''

    # load data
    ds = pd.read_csv(test_data_path)

    # make masks for all the transcripts
    mask = inputs != -100.0

    # obtain lengths of all the transcripts
    lengths = np.sum(mask, axis=1)

    # convert to lists and remove padding + first cond token
    preds = preds.tolist()
    preds = [pred[1:lengths[i]] for i, pred in enumerate(preds)]

    labels = labels.tolist()
    labels = [label[:lengths[i]-1] for i, label in enumerate(labels)]

    inputs = inputs.tolist()
    inputs = [input[:lengths[i]] for i, input in enumerate(inputs)]

    # get the condition for each of the samples
    # take the first token to get the condition
    condition_samples = []
    for i in range(len(inputs)):
        condition_samples.append(inputs[i][0])

    labels_ctrl = []
    genes = []
    transcripts = []
    sequences_ds = []

    sequence_list = list(ds['sequence'])
    ctrl_sequence_list_ds = list(ds['ctrl_sequence'])
    genes_list_ds = list(ds['gene'])
    transcripts_list_ds = list(ds['transcript'])
    condition_list = list(ds['condition'])
    codon_sequences = []

    # for loop which takes the original transcript one-hot sequence and converts into the [cond_val, x] version (stored in sequences_ds) 
    for i in range(len(sequence_list)):
        x = sequence_list[i][1:-1].split(', ')
        x = [int(i) for i in x]
        cond_val = condition_values[condition_list[i]]
        # get codon sequence from x
        codon_seq = [id_to_codon[i] for i in x]
        # convert to string
        codon_seq = ''.join(codon_seq)
        codon_sequences.append(codon_seq)
        # get the remainder
        # prepend the condition value to the sequence
        x = np.insert(x, 0, cond_val)
        sequences_ds.append(x)

    # for loop to get the control label for all the transcripts
    for i in range(len(inputs)):
        condition_sample = inverse_condition_values[condition_samples[i]]
        # search for inputs[i] in sequences_ds get index
        for j in range(len(sequences_ds)):
            if checkArrayEquality(sequences_ds[j], inputs[i]) and condition_sample == condition_list[j]:
                index = j
                break

        ctrl_sample = ctrl_sequence_list_ds[index]
        ctrl_sample = ctrl_sample[1:-1].split(', ')
        ctrl_sample = [float(k) for k in ctrl_sample]
        labels_ctrl.append(ctrl_sample)
        genes.append(genes_list_ds[index])
        transcripts.append(transcripts_list_ds[index])

    # Model output: the first dim of pred is ctrl, second is depr difference
    # process ctrl predictions
    ctrl_preds = []
    for i in range(len(preds)):
        pred_sample = preds[i]
        pred_sample = np.asarray(pred_sample)
        # get first dim
        pred_sample = pred_sample[:, 0]
        ctrl_preds.append(pred_sample)

    # process depr difference predictions
    depr_diffs = []
    for i in range(len(preds)):
        pred_sample = preds[i]
        pred_sample = np.asarray(pred_sample)
        # get second dim
        pred_sample = pred_sample[:, 1]
        depr_diffs.append(pred_sample)

    # obtain the full predictions: the summation of the ctrl and depr difference predictions
    full_preds = []
    for i in range(len(preds)):
        full_preds.append(ctrl_preds[i] + depr_diffs[i])

    # np log 1 + x the labels
    labels_ctrl = [np.log1p(label) for label in labels_ctrl]

    # labels depr difference
    labels_depr_diff = []
    for i in range(len(labels)):
        labels_depr_diff.append(np.asarray(labels[i]) - np.asarray(labels_ctrl[i]))

    # plot ten best samples
    # get pearson corr for each sample
    pearson_corrs = []
    mae_list = []
    mape_list = []
    for i in range(len(full_preds)):
        pearson_corrs.append(pearson_mask(full_preds[i], labels[i]))
        mae_list.append(mae_mask(full_preds[i], labels[i]))
        mape_list.append(mape_mask(full_preds[i], labels[i]))

    # pearson mean for each condition
    pearson_means = [[] for i in range(6)]
    for i in range(len(pearson_corrs)):
        pearson_means[condition_samples[i]-64].append(pearson_corrs[i])
    
    # print means
    print("Pearson Correlation Coefficient Means - Per Condition")
    for i in range(len(pearson_means)):
        print("Condition: ", inverse_condition_values[i+64], " Mean: ", np.mean(pearson_means[i]), " Std: ", np.std(pearson_means[i]), " Num Samples: ", len(pearson_means[i]))

    conds_colors = ["#264653", "#2A9D8F", "#E9C46A", '#F4A261', '#E76F51', '#ef476f']
    # make distribution plot for each condition
    for i in range(len(pearson_means)):
        # sns histogram
        sns.histplot(pearson_means[i], color=conds_colors[i], kde=True, bins=100)
        plt.title("Pearson Correlation Coefficient Distribution - " + inverse_condition_values[i+64])
        plt.xlabel("Pearson Correlation Coefficient")
        plt.ylabel("Frequency")
        plt.savefig(output_loc + "/condition_dists/pearson_" + inverse_condition_values[i+64] + ".png")
        plt.show()
        plt.clf()

    pearson_corrs_ctrl = []
    for i in range(len(ctrl_preds)):
        pearson_corrs_ctrl.append(pearson_mask(ctrl_preds[i], labels_ctrl[i]))

    # output all the predictions into df from lists
    output_analysis_df = pd.DataFrame(list(zip(transcripts, genes, codon_sequences, pearson_corrs, pearson_corrs_ctrl, mae_list, mape_list, condition_list)), columns =['Transcript', 'Gene', 'Sequence', 'Full Prediction Pearson Correlation', 'Control Prediction Pearson Correlation', 'MAE', 'MAPE', 'Deprivation Condition'])
    output_analysis_df.to_csv(output_loc + "/analysis.csv", index=False)

    print("Saved model prediction outputs file to ", output_loc + "/analysis.csv")

    print("#"*20)

    num_plots = 10

    ######### 
    # plot (num_plots) best samples
    #########
    best_samples = sorted(range(len(pearson_corrs)), key = lambda sub: pearson_corrs[sub])[-num_plots:]
    # print best pearson corrs
    print("List of best pearson correlations: ", [pearson_corrs[i] for i in best_samples])

    print("#"*20)

    # for i in range(num_plots):
    #     out_loc = output_loc + "/full_plots/sample_" + str(best_samples[i]) + '_' + str(inverse_condition_values[condition_samples[best_samples[i]]]) + "_best_" + transcripts[best_samples[i]] + "_" + genes[best_samples[i]] + ".png"
    #     make_plot(full_preds[best_samples[i]], depr_diffs[best_samples[i]], ctrl_preds[best_samples[i]], labels[best_samples[i]], labels_ctrl[best_samples[i]], labels_depr_diff[best_samples[i]], out_loc, transcripts[best_samples[i]], genes[best_samples[i]], inverse_condition_values[condition_samples[best_samples[i]]])

    ######### 
    # plot (num_plots) worst samples
    #########
    worst_samples = sorted(range(len(pearson_corrs)), key = lambda sub: pearson_corrs[sub])[:num_plots]
    
    # print worst pearson corrs
    print("Worst Pearson Correlations: ", [pearson_corrs[i] for i in worst_samples])

    print("#"*20)

    for i in range(num_plots):
        out_loc = output_loc + "/full_plots/sample_" + str(worst_samples[i]) + '_' + str(inverse_condition_values[condition_samples[worst_samples[i]]]) + "_worst_" + transcripts[worst_samples[i]] + "_" + genes[worst_samples[i]] + ".png"
        make_plot(full_preds[worst_samples[i]], depr_diffs[worst_samples[i]], ctrl_preds[worst_samples[i]], labels[worst_samples[i]], labels_ctrl[worst_samples[i]], labels_depr_diff[worst_samples[i]], out_loc, transcripts[worst_samples[i]], genes[worst_samples[i]], inverse_condition_values[condition_samples[worst_samples[i]]])

def quantile_metric(preds, labels, inputs, output_loc, test_data_path):
    # make mask removing those that have a input of -100
    mask = inputs != -100.0

    # get lengths of each sequence
    lengths = np.sum(mask, axis=1)

    # convert to lists and remove padding
    preds = preds.tolist()
    labels = labels.tolist()
    inputs = inputs.tolist()

    preds = [pred[:lengths[i]] for i, pred in enumerate(preds)]
    labels = [label[:lengths[i]] for i, label in enumerate(labels)]
    inputs = [input[:lengths[i]] for i, input in enumerate(inputs)]

    condition_samples = []

    # get conditions for each sample
    # take cond_val to get the condition
    for i in range(len(inputs)):
        condition_samples.append(inputs[i][0])

    genes = []
    transcripts = []

    ds = pd.read_csv(test_data_path)

    sequences_ds = []

    sequence_list = list(ds['sequence'])
    genes_list_ds = list(ds['gene'])
    transcripts_list_ds = list(ds['transcript'])
    condition_list = list(ds['condition'])
    codon_sequences = []

    for i in range(len(sequence_list)):
        x = sequence_list[i][1:-1].split(', ')
        x = [int(i) for i in x]
        cond_val = condition_values[condition_list[i]]
        # get codon sequence from x
        codon_seq = [id_to_codon[i] for i in x]
        # convert to string
        codon_seq = ''.join(codon_seq)
        codon_sequences.append(codon_seq)
        # get the remainder
        add_val = (cond_val) * 64
        x = [i + add_val for i in x]
        sequences_ds.append(x)

    for i in range(len(inputs)):
        condition_sample = inverse_condition_values[condition_samples[i]]
        # search for inputs[i] in sequences_ds get index
        for j in range(len(sequences_ds)):
            if sequences_ds[j] == inputs[i] and condition_sample == condition_list[j]:
                index = j
                break

        genes.append(genes_list_ds[index])
        transcripts.append(transcripts_list_ds[index])

    # ctrl predictions
    ctrl_preds = []
    for i in range(len(preds)):
        pred_sample = preds[i]
        pred_sample = np.asarray(pred_sample)
        # get first dim
        pred_sample = pred_sample[:, 0]
        ctrl_preds.append(pred_sample)

    depr_diffs = []
    for i in range(len(preds)):
        pred_sample = preds[i]
        pred_sample = np.asarray(pred_sample)
        # get second dim
        pred_sample = pred_sample[:, 1]
        depr_diffs.append(pred_sample)

    full_preds = []
    for i in range(len(preds)):
        full_preds.append(ctrl_preds[i] + depr_diffs[i])
        # print(len(full_preds[i]), len(labels[i]))

    # plot ten best samples
    # get pearson corr for each sample
    pearson_corrs = []
    for i in range(len(full_preds)):
        pearson_corrs.append(pearson_mask(full_preds[i], labels[i]))

    # go through each prediction
    quantiles = [j*0.1 for j in range(10)]
    all_f1_scores = []
    all_prec_scores = []
    all_recall_scores = []
    for i in range(len(pearson_corrs)):
        # for each sample do metric for each quantile
        # 10 quantiles
        f1_sample = []  # f1 score for each sample
        prec_sample = []  # precision score for each sample
        recall_sample = []  # recall score for each sample
        # iterate through each quantile
        for k in range(len(quantiles)): # label quantile val is nan
            pred_quantile_val = np.quantile(full_preds[i], quantiles[k])
            label_quantile_val = np.nanquantile(labels[i], quantiles[k])

            # binarize the prediction and label based on the quantile including the nans
            # copy the prediction
            pred_quantile = np.copy(full_preds[i])
            # set all values below the quantile to 0
            pred_quantile[pred_quantile < pred_quantile_val] = 0
            # set all values above the quantile to 1
            pred_quantile[pred_quantile >= pred_quantile_val] = 1
            
            # copy the label
            label_quantile = np.copy(labels[i])
            # set all values below the quantile to 0
            label_quantile[label_quantile < label_quantile_val] = 0
            # set all values above the quantile to 1
            label_quantile[label_quantile >= label_quantile_val] = 1

            # get f1 score for the pred_quantile and label_quantile
            # print("label quantile val:", label_quantile_val)
            # print("pred quantile val:", pred_quantile_val)
            # print(pred_quantile, label_quantile)
            # print("TO FUNCTION")
            f1_val = f1_score_masked(pred_quantile, label_quantile)
            prec_val = prec_score_masked(pred_quantile, label_quantile)
            recall_val = recall_score_masked(pred_quantile, label_quantile)

            # print(f1_val)

            f1_sample.append(f1_val)
            prec_sample.append(prec_val)
            recall_sample.append(recall_val)

        all_f1_scores.append(f1_sample)
        all_prec_scores.append(prec_sample)
        all_recall_scores.append(recall_sample)

    # make stats for each quantile from all samples
    all_f1_scores = np.asarray(all_f1_scores)
    all_f1_scores = np.transpose(all_f1_scores)

    all_prec_scores = np.asarray(all_prec_scores)
    all_prec_scores = np.transpose(all_prec_scores)

    all_recall_scores = np.asarray(all_recall_scores)
    all_recall_scores = np.transpose(all_recall_scores)

    # print(all_f1_scores.shape)

    # get mean and std for each quantile
    quantile_means_f1 = []
    quantile_stds_f1 = []
    quantile_means_prec = []
    quantile_stds_prec = []
    quantile_means_recall = []
    quantile_stds_recall = []
    for f in range(len(all_f1_scores)):
        quantile_means_f1.append(np.mean(all_f1_scores[f]))
        quantile_stds_f1.append(np.std(all_f1_scores[f]))

        quantile_means_prec.append(np.mean(all_prec_scores[f]))
        quantile_stds_prec.append(np.std(all_prec_scores[f]))

        quantile_means_recall.append(np.mean(all_recall_scores[f]))
        quantile_stds_recall.append(np.std(all_recall_scores[f]))

    # plot the mean and std for each quantile
    sns.set_style("whitegrid")
    plt.errorbar(quantiles, quantile_means_f1, yerr=quantile_stds_f1, fmt='o', color='#130f40')
    # draw a smooth curve connecting the points from means
    X_Y_Spline = make_interp_spline(quantiles, quantile_means_f1)
    # Returns evenly spaced numbers
    # over a specified interval.
    X_ = np.linspace(min(quantiles), max(quantiles), 500)
    Y_ = X_Y_Spline(X_)
    plt.plot(X_, Y_, color='#0097e6')
    # shade the area under the curve
    plt.fill_between(X_, Y_, color='#0097e6', alpha=0.4)
    plt.xlabel('Quantile')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Quantile')
    plt.savefig(output_loc + 'f1_quantile.png')
    plt.show()
    plt.clf()

    sns.set_style("whitegrid")
    plt.errorbar(quantiles, quantile_means_prec, yerr=quantile_stds_prec, fmt='o', color='#130f40')
    # draw a smooth curve connecting the points from means
    X_Y_Spline = make_interp_spline(quantiles, quantile_means_prec)
    # Returns evenly spaced numbers
    # over a specified interval.
    X_ = np.linspace(min(quantiles), max(quantiles), 500)
    Y_ = X_Y_Spline(X_)
    plt.plot(X_, Y_, color='#0097e6')
    # shade the area under the curve
    plt.fill_between(X_, Y_, color='#0097e6', alpha=0.4)
    plt.xlabel('Quantile')
    plt.ylabel('Precision Score')
    plt.title('Precision Score vs Quantile')
    plt.savefig(output_loc + 'prec_quantile.png')
    plt.show()
    plt.clf()

    sns.set_style("whitegrid")
    plt.errorbar(quantiles, quantile_means_recall, yerr=quantile_stds_recall, fmt='o', color='#130f40')
    # draw a smooth curve connecting the points from means
    X_Y_Spline = make_interp_spline(quantiles, quantile_means_recall)
    # Returns evenly spaced numbers
    # over a specified interval.
    X_ = np.linspace(min(quantiles), max(quantiles), 500)
    Y_ = X_Y_Spline(X_)
    plt.plot(X_, Y_, color='#0097e6')
    # shade the area under the curve
    plt.fill_between(X_, Y_, color='#0097e6', alpha=0.4)
    plt.xlabel('Quantile')
    plt.ylabel('Recall Score')
    plt.title('Recall Score vs Quantile')
    plt.savefig(output_loc + 'recall_quantile.png')
    plt.show()
    plt.clf()

    # using quantile means get the area under the curve
    auc_f1 = np.trapz(quantile_means_f1, dx=0.1)
    auc_prec = np.trapz(quantile_means_prec, dx=0.1)
    auc_recall = np.trapz(quantile_means_recall, dx=0.1)
    print("AUC F1 Score: ", auc_f1)
    print("AUC Precision Score: ", auc_prec)
    print("AUC Recall Score: ", auc_recall)

# %%
# attention maps interpretability functions
def attention_maps(model, test_dataset, output_loc, cond_attr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    list_attn_matrices = []
    max_len = 0
    lens_list = []
    with torch.no_grad():
        for i, (x, y, ctrl_y, gene, transcript) in enumerate(test_dataset):
            # number of zero elements in y
            # get condition from x 
            condition = inverse_condition_values[(x // 64)[0].item()]

            if condition == cond_attr or cond_attr == 'ALL':
                y_zero = np.nan_to_num(y)
                num_zeros = len(y) - np.count_nonzero(y_zero)

                lengths = torch.tensor([len(x)])

                x = torch.tensor(x).unsqueeze(0)
                y = torch.tensor(y).unsqueeze(0)
                ctrl_y = torch.tensor(ctrl_y).unsqueeze(0)
                
                x = pad_sequence(x, batch_first=True, padding_value=192) 
                y = pad_sequence(y, batch_first=True, padding_value=-1)
                ctrl_y = pad_sequence(ctrl_y, batch_first=True, padding_value=-1)

                out_batch = {}

                out_batch["input_ids"] = x
                out_batch["labels"] = y
                out_batch["lengths"] = lengths
                out_batch["labels_ctrl"] = ctrl_y

                # send batch to device
                for k, v in out_batch.items():
                    out_batch[k] = v.to(device)

                out = model(out_batch["input_ids"], output_attentions = True, return_dict = True)
                attn_vec1 = out.attentions[0].cpu().detach().numpy()
                attn_vec2 = out.attentions[1].cpu().detach().numpy()
                attn_vec3 = out.attentions[2].cpu().detach().numpy()

                attn_vec_full = attn_vec1 # only first layer because this is the only one that looks at the input

                # remove dim 0
                attn_vec_full = np.squeeze(attn_vec_full, axis=0)

                # average across heads
                attn_vec_full = np.mean(attn_vec_full, axis=0)

                list_attn_matrices.append(attn_vec_full)

                if len(attn_vec_full) > max_len:
                    max_len = len(attn_vec_full)
                
                lens_list.append(len(attn_vec_full))

    mean_len = int(np.mean(lens_list))
    # make matrix of shape max_len x max_len
    attn_matrix = np.zeros((max_len, max_len))
    # add all attention matrices to the matrix
    count_lengths = np.zeros(max_len)
    for i in range(len(list_attn_matrices)):
        count_lengths[:len(list_attn_matrices[i])] += 1

    for i in range(len(list_attn_matrices)):
        attn_matrix[:len(list_attn_matrices[i]), :len(list_attn_matrices[i])] += (list_attn_matrices[i] / count_lengths[:len(list_attn_matrices[i])])

    # crop the matrix to the mean length
    attn_matrix = attn_matrix[:mean_len, :mean_len]

    # plot the attention matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(attn_matrix)
    plt.savefig(output_loc + 'attention_map_' + str(cond_attr) + '.png')
    plt.show()
    plt.clf()

    # make dataframe position_to_Asite and the attention weight for that position
    pos_A_site_df = []
    attn_weight_df = []
    for i in range(len(attn_matrix)):
        for j in range(len(attn_matrix)):
            pos_A_site_df.append(j-i)
            attn_weight_df.append(attn_matrix[i][j])

    # make dataframe
    df_attn_weights = pd.DataFrame({'pos_A_site': pos_A_site_df, 'attn_weight': attn_weight_df})

    # make a lineplot from the dataframe using seaborn
    plt.figure(figsize=(10,10))
    sns.lineplot(x='pos_A_site', y='attn_weight', data=df_attn_weights, errorbar="sd")
    plt.savefig(output_loc + 'amaps_crop_lineplot_sns_xlimfull_' + str(cond_attr) + '.png')
    plt.show()
    plt.clf()

    # make a lineplot from the dataframe using seaborn
    plt.figure(figsize=(10,10))
    sns.lineplot(x='pos_A_site', y='attn_weight', data=df_attn_weights, errorbar="sd")
    # set x scale to be from -20 to 20
    plt.xlim(-20, 20)
    plt.savefig(output_loc + 'amaps_crop_lineplot_sns_xlim20_' + str(cond_attr) + '.png')
    plt.show()
    plt.clf()

    # log transform the attention maps
    attention_maps_logged = np.log(attn_matrix)

    # plot the log transformed attention matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(attention_maps_logged)
    plt.savefig(output_loc + 'attention_map_logged_' + str(cond_attr) + '.png')
    plt.show()
    plt.clf()

    # make dataframe position_to_Asite and the attention weight for that position
    pos_A_site_df_log = []
    attn_weight_df_log = []
    for i in range(len(attention_maps_logged)):
        for j in range(len(attention_maps_logged)):
            pos_A_site_df_log.append(j-i)
            attn_weight_df_log.append(attention_maps_logged[i][j])

    # make dataframe
    df_attn_weights_log = pd.DataFrame({'pos_A_site': pos_A_site_df_log, 'attn_weight': attn_weight_df_log})

     # make a lineplot from the dataframe using seaborn
    plt.figure(figsize=(10,10))
    sns.lineplot(x='pos_A_site', y='attn_weight', data=df_attn_weights_log, errorbar="sd")
    plt.savefig(output_loc + 'amaps_crop_logged_lineplot_sns_xlimfull_' + str(cond_attr) + '.png')
    plt.show()
    plt.clf()

    # make a lineplot from the dataframe using seaborn
    plt.figure(figsize=(10,10))
    sns.lineplot(x='pos_A_site', y='attn_weight', data=df_attn_weights_log, errorbar="sd")
    plt.xlim(-20, 20)
    plt.savefig(output_loc + 'amaps_crop_logged_lineplot_sns_xlim20_' + str(cond_attr) + '.png')
    plt.show()
    plt.clf()

    # save the attention matrix
    np.save(output_loc + 'attention_map_' + str(cond_attr) + '.npy', attn_matrix)

    # check how many elements next to the principal diagonal have high values on average
    left_diag = []
    right_diag = []

    threshold_val = 3

    attention_maps_logged_thresholded = [] 

    queue_ribo_dist_start = []
    queue_ribo_dist_end = []

    for i in range(attention_maps_logged.shape[0]):
        # A site value
        max_val_i = np.max(attention_maps_logged[i])
        max_val_i_index = np.argmax(attention_maps_logged[i])

        # left elements
        left_i = attention_maps_logged[i, :i]
        # number of elements with values between max_val_i and max_val_i - threshold_val
        left_i_threshold = left_i > (max_val_i - threshold_val)
        # sum of non zero elements
        left_diag.append(np.sum(left_i_threshold))

        # right elements
        right_i = attention_maps_logged[i, i+1:]
        # number of elements with values between max_val_i and max_val_i - threshold_val
        right_i_threshold = right_i > (max_val_i - threshold_val)
        right_diag.append(np.sum(right_i_threshold))

        # row of the thresholded attention map 
        row = np.concatenate((left_i_threshold, [1], right_i_threshold))
        attention_maps_logged_thresholded.append(row)

        # find the distance of the queueing ribosome using the row vector of the thresholded attention map
        # find the index of the element with value 1 behind i, such that it is after the small block of 0 
        queue_ribo_dist_start.append(i - np.argmax(row))

    # print means of left and right diagonals
    print("Number of important codons to the left (avg): ", np.mean(left_diag))
    print("Number of important codons to the right (avg): ", np.mean(right_diag))
    # get the mode of the queueing ribosome distances
    print("Average distance of the queued ribosomes from the A site: ", stats.mode(queue_ribo_dist_start)[0], " codons, ", " (", (stats.mode(queue_ribo_dist_start)[1] / len(queue_ribo_dist_start)) * 100, " percent of the times)")

    attention_maps_logged_thresholded = np.array(attention_maps_logged_thresholded)

    # plot the log transformed attention matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(attention_maps_logged_thresholded)
    plt.savefig(output_loc + 'attention_map_logged_Thresh3_' + str(cond_attr) + '.png')  
    plt.show()
    plt.clf()


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def captum_LayerGradAct(model, test_dataset, output_loc):
    model_fin = model_finalexp(model)
        
    lig = LayerGradientXActivation(model_fin, model_fin.model.transformer.word_embedding)

    attributions_total = []
    lens_list = []
    indices_main_list = []

    # set torch graph to allow unused tensors
    with torch.autograd.set_detect_anomaly(True):
        for i, (x, y, _, _, _) in enumerate(test_dataset):
            x = torch.tensor(x).unsqueeze(0)
            
            x = pad_sequence(x, batch_first=True, padding_value=192) 

            out_batch = {}

            out_batch["input_ids"] = x
            
            out_batch["input_ids"] = torch.tensor(out_batch["input_ids"]).to(device).to(torch.int32)

            # get indices of top 95% of values in y 
            quantile_95 = np.nanquantile(y, 0.95)
            indices = np.where(y > quantile_95)[0]

            # indices append
            for k in indices:
                indices_main_list.append(k)
            
            # make len(x[0]) x len(x[0]) matrix
            len_sample = len(x[0])
            attributions_sample = np.zeros((len_sample, len_sample))

            for j in indices:
                index_val = j

                index_val = torch.tensor(index_val).to(device)

                attributions = lig.attribute((out_batch["input_ids"]), additional_forward_args=index_val)
                attributions = attributions.squeeze(1)
                attributions = torch.sum(attributions, dim=1)
                attributions = attributions / torch.norm(attributions)
                attributions = attributions.detach().cpu().numpy()
                attributions_sample[j] = attributions
            
            attributions_sample = np.array(attributions_sample)
            print(i, attributions_sample.shape)
            attributions_total.append(attributions_sample)
            lens_list.append(len_sample)

    # save all the attributions
    model_name = output_loc.split('/')[-4]
    attr_save_loc = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Darnell_Full/Captum/' + model_name + '/'
    # make directory if it doesn't exist
    if not os.path.exists(attr_save_loc):
        os.makedirs(attr_save_loc)

    np.save(attr_save_loc + 'ALL_Attributions_LayerGradxAct.npy', attributions_total)

    max_len = max(lens_list)
    # make matrix of shape max_len x max_len
    attr_matrix = np.zeros((max_len, max_len))
    # add all attention matrices to the matrix
    for i in range(len(attributions_total)):
        attr_matrix[:len(attributions_total[i]), :len(attributions_total[i])] += attributions_total[i]

    # div each index by the count of the index in the indices_main_list
    for i in range(len(attr_matrix)):
        count_i = np.count_nonzero(indices_main_list == i)
        if count_i != 0:
            attr_matrix[i] /= count_i

    # trim the matrix to the mean of the lengths
    attr_matrix = attr_matrix[:int(np.mean(lens_list)), :int(np.mean(lens_list))]

    # save the matrix
    np.save(output_loc + "LGAct_matrix.npy", attr_matrix)

    # plot the matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(attr_matrix)
    plt.savefig(output_loc + 'LGXAct_Sum.png')
    plt.show()
    plt.clf()

    # make dataframe position_to_Asite and the attention weight for that position
    pos_A_site_df = []
    attr_weight_df = []
    for i in range(len(attr_matrix)):
        for j in range(len(attr_matrix)):
            pos_A_site_df.append(j-i)
            attr_weight_df.append(attr_matrix[i][j])

    # make dataframe
    df_attn_weights = pd.DataFrame({'pos_A_site': pos_A_site_df, 'attn_weight': attr_weight_df})

    # make a lineplot from the dataframe using seaborn
    plt.figure(figsize=(10,10))
    sns.lineplot(x='pos_A_site', y='attn_weight', data=df_attn_weights, errorbar="sd")
    plt.savefig(output_loc + 'LGXAct_crop_lineplot_sns_xlimfull.png')
    plt.show()
    plt.clf()

    # make a lineplot from the dataframe using seaborn
    plt.figure(figsize=(10,10))
    sns.lineplot(x='pos_A_site', y='attn_weight', data=df_attn_weights, errorbar="sd")
    # set x scale to be from -20 to 20
    plt.xlim(-20, 20)
    plt.savefig(output_loc + 'LGXAct_crop_lineplot_sns_xlim20.png')
    plt.show()
    plt.clf()

    # log the matrix 
    attr_matrix_logged = np.log10(attr_matrix)

    # set nans to -10
    attr_matrix_logged[np.isnan(attr_matrix_logged)] = -7

    # # plot the logged matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(attr_matrix_logged)
    plt.savefig(output_loc + 'LGXAct_Sum_Logged.png')
    plt.show()
    plt.clf()

    # make dataframe position_to_Asite and the attention weight for that position
    pos_A_site_df_log = []
    attr_weight_df_log = []
    for i in range(len(attr_matrix_logged)):
        for j in range(len(attr_matrix_logged)):
            pos_A_site_df_log.append(j-i)
            attr_weight_df_log.append(attr_matrix_logged[i][j])

    # make dataframe
    df_attn_weights_log = pd.DataFrame({'pos_A_site': pos_A_site_df_log, 'attn_weight': attr_weight_df_log})

     # make a lineplot from the dataframe using seaborn
    plt.figure(figsize=(10,10))
    sns.lineplot(x='pos_A_site', y='attn_weight', data=df_attn_weights_log, errorbar="sd")
    plt.savefig(output_loc + 'LGXAct_crop_logged_lineplot_sns_xlimfull.png')
    plt.show()
    plt.clf()

    # make a lineplot from the dataframe using seaborn
    plt.figure(figsize=(10,10))
    sns.lineplot(x='pos_A_site', y='attn_weight', data=df_attn_weights_log, errorbar="sd")
    plt.xlim(-20, 20)
    plt.savefig(output_loc + 'LGXAct_crop_logged_lineplot_sns_xlim20.png')
    plt.show()
    plt.clf()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def captum_LayerGrad(model, test_dataset, output_loc):
    model_fin = model_finalexpLIG(model)
        
    lig = LayerIntegratedGradients(model_fin, model_fin.model.transformer.word_embedding)

    attributions_total = []
    lens_list = []
    indices_main_list = []

    # set torch graph to allow unused tensors
    with torch.autograd.set_detect_anomaly(True):
        for i, (x, y, _, _, _) in enumerate(test_dataset):
            x = torch.tensor(x)
            
            # x = pad_sequence(x, batch_first=True, padding_value=192) 

            out_batch = {}

            out_batch["input_ids"] = x
            
            out_batch["input_ids"] = torch.tensor(out_batch["input_ids"]).to(device).to(torch.int32)

            baseline_inp = torch.ones(out_batch["input_ids"].shape) * 192
            baseline_inp = baseline_inp.to(device).to(torch.int32)

            # get indices of top 95% of values in y 
            quantile_95 = np.nanquantile(y, 0.95)
            indices = np.where(y > quantile_95)[0]

            # indices append
            for k in indices:
                indices_main_list.append(k)
            
            # make len(x[0]) x len(x[0]) matrix
            len_sample = len(x)
            attributions_sample = np.zeros((len_sample, len_sample))

            for j in indices:
                index_val = j

                index_val = torch.tensor(index_val).to(device)

                attributions, approximation_error = lig.attribute((out_batch["input_ids"]), baselines=baseline_inp, 
                                                        method = 'gausslegendre', return_convergence_delta = True, additional_forward_args=index_val, n_steps=20, internal_batch_size=2048)

                
                attributions = attributions.squeeze(1)
                attributions = torch.sum(attributions, dim=1)
                attributions = attributions / torch.norm(attributions)
                attributions = attributions.detach().cpu().numpy()
                attributions_sample[j] = attributions
            
            attributions_sample = np.array(attributions_sample)
            print(i, attributions_sample.shape)
            attributions_total.append(attributions_sample)
            lens_list.append(len_sample)

    # save all the attributions
    model_name = output_loc.split('/')[-4]
    attr_save_loc = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Darnell_Full/Captum/' + model_name + '/'
    # make directory if it doesn't exist
    if not os.path.exists(attr_save_loc):
        os.makedirs(attr_save_loc)

    np.save(attr_save_loc + 'ALL_Attributions_LIG.npy', attributions_total)

    max_len = max(lens_list)
    # make matrix of shape max_len x max_len
    attr_matrix = np.zeros((max_len, max_len))
    # add all attention matrices to the matrix
    for i in range(len(attributions_total)):
        attr_matrix[:len(attributions_total[i]), :len(attributions_total[i])] += attributions_total[i]

    # div each index by the count of the index in the indices_main_list
    for i in range(len(attr_matrix)):
        count_i = np.count_nonzero(indices_main_list == i)
        if count_i != 0:
            attr_matrix[i] /= count_i

    # trim the matrix to the mean of the lengths
    attr_matrix = attr_matrix[:int(np.mean(lens_list)), :int(np.mean(lens_list))]

    # save the matrix
    np.save(output_loc + "LIG_matrix.npy", attr_matrix)

    # plot the matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(attr_matrix)
    plt.savefig(output_loc + 'LIG_Sum.png')
    plt.show()
    plt.clf()

    # make dataframe position_to_Asite and the attention weight for that position
    pos_A_site_df = []
    attr_weight_df = []
    for i in range(len(attr_matrix)):
        for j in range(len(attr_matrix)):
            pos_A_site_df.append(j-i)
            attr_weight_df.append(attr_matrix[i][j])

    # make dataframe
    df_attn_weights = pd.DataFrame({'pos_A_site': pos_A_site_df, 'attn_weight': attr_weight_df})

    # make a lineplot from the dataframe using seaborn
    plt.figure(figsize=(10,10))
    sns.lineplot(x='pos_A_site', y='attn_weight', data=df_attn_weights, errorbar="sd")
    plt.savefig(output_loc + 'LIG_crop_lineplot_sns_xlimfull.png')
    plt.show()
    plt.clf()

    # make a lineplot from the dataframe using seaborn
    plt.figure(figsize=(10,10))
    sns.lineplot(x='pos_A_site', y='attn_weight', data=df_attn_weights, errorbar="sd")
    # set x scale to be from -20 to 20
    plt.xlim(-20, 20)
    plt.savefig(output_loc + 'LIG_crop_lineplot_sns_xlim20.png')
    plt.show()
    plt.clf()

    # log the matrix 
    attr_matrix_logged = np.log10(attr_matrix)

    # set nans to -10
    attr_matrix_logged[np.isnan(attr_matrix_logged)] = -7

    # # plot the logged matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(attr_matrix_logged)
    plt.savefig(output_loc + 'LIG_Sum_Logged.png')
    plt.show()
    plt.clf()

    # make dataframe position_to_Asite and the attention weight for that position
    pos_A_site_df_log = []
    attr_weight_df_log = []
    for i in range(len(attr_matrix_logged)):
        for j in range(len(attr_matrix_logged)):
            pos_A_site_df_log.append(j-i)
            attr_weight_df_log.append(attr_matrix_logged[i][j])

    # make dataframe
    df_attn_weights_log = pd.DataFrame({'pos_A_site': pos_A_site_df_log, 'attn_weight': attr_weight_df_log})

     # make a lineplot from the dataframe using seaborn
    plt.figure(figsize=(10,10))
    sns.lineplot(x='pos_A_site', y='attn_weight', data=df_attn_weights_log, errorbar="sd")
    plt.savefig(output_loc + 'LIG_crop_logged_lineplot_sns_xlimfull.png')
    plt.show()
    plt.clf()

    # make a lineplot from the dataframe using seaborn
    plt.figure(figsize=(10,10))
    sns.lineplot(x='pos_A_site', y='attn_weight', data=df_attn_weights_log, errorbar="sd")
    plt.xlim(-20, 20)
    plt.savefig(output_loc + 'LIG_crop_logged_lineplot_sns_xlim20.png')
    plt.show()
    plt.clf()

# %%
def interpretability_panels(model, preds, labels, inputs, output_loc, test_data_path):
    '''
    panel for interpretability
    1. labels
    2. prediction of the model
    3. attention maps
    4. layer grad x activation
    5. layer integrated gradients
    '''
    # load data
    ds = pd.read_csv(test_data_path)

    # make masks for all the transcripts
    mask = inputs != -100.0

    # obtain lengths of all the transcripts
    lengths = np.sum(mask, axis=1)

    # convert to lists and remove padding
    preds = preds.tolist()
    preds = [pred[1:lengths[i]] for i, pred in enumerate(preds)]

    labels = labels.tolist()
    labels = [label[:lengths[i]-1] for i, label in enumerate(labels)]

    inputs = inputs.tolist()
    inputs = [input[:lengths[i]] for i, input in enumerate(inputs)]

    # get the condition for each of the samples
    condition_samples = []
    for i in range(len(inputs)):
        condition_samples.append(inputs[i][0])

    labels_ctrl = []
    genes = []
    transcripts = []
    sequences_ds = []

    sequence_list = list(ds['sequence'])
    ctrl_sequence_list_ds = list(ds['ctrl_sequence'])
    genes_list_ds = list(ds['gene'])
    transcripts_list_ds = list(ds['transcript'])
    condition_list = list(ds['condition'])
    codon_sequences = []

    # for loop which takes the original transcript one-hot sequence and converts into the (cond_val, orig seq) (stored in sequences_ds) 
    for i in range(len(sequence_list)):
        x = sequence_list[i][1:-1].split(', ')
        x = [int(i) for i in x]
        cond_val = condition_values[condition_list[i]]
        # get codon sequence from x
        codon_seq = [id_to_codon[i] for i in x]
        # convert to string
        codon_seq = ''.join(codon_seq)
        codon_sequences.append(codon_seq)
        # get the remainder
        # prepend the condition value to the sequence
        x = np.insert(x, 0, cond_val)
        sequences_ds.append(x)

    # for loop to get the control label for all the transcripts
    for i in range(len(inputs)):
        condition_sample = inverse_condition_values[condition_samples[i]]
        # search for inputs[i] in sequences_ds get index
        for j in range(len(sequences_ds)):
            if checkArrayEquality(sequences_ds[j], inputs[i]) and condition_sample == condition_list[j]:
                index = j
                break

        ctrl_sample = ctrl_sequence_list_ds[index]
        ctrl_sample = ctrl_sample[1:-1].split(', ')
        ctrl_sample = [float(k) for k in ctrl_sample]
        labels_ctrl.append(ctrl_sample)
        genes.append(genes_list_ds[index])
        transcripts.append(transcripts_list_ds[index])

    # Model output: the first dim of pred is ctrl, second is depr difference
    # process ctrl predictions
    ctrl_preds = []
    for i in range(len(preds)):
        pred_sample = preds[i]
        pred_sample = np.asarray(pred_sample)
        # get first dim
        pred_sample = pred_sample[:, 0]
        ctrl_preds.append(pred_sample)

    # process depr difference predictions
    depr_diffs = []
    for i in range(len(preds)):
        pred_sample = preds[i]
        pred_sample = np.asarray(pred_sample)
        # get second dim
        pred_sample = pred_sample[:, 1]
        depr_diffs.append(pred_sample)

    # obtain the full predictions: the summation of the ctrl and depr difference predictions
    full_preds = []
    for i in range(len(preds)):
        full_preds.append(ctrl_preds[i] + depr_diffs[i])

    # np log 1 + x the labels
    labels_ctrl = [np.log1p(label) for label in labels_ctrl]

    # labels depr difference
    labels_depr_diff = []
    for i in range(len(labels)):
        labels_depr_diff.append(np.asarray(labels[i]) - np.asarray(labels_ctrl[i]))

    # calculate the variance in the depr diff labels
    var_labels_depr_diff = []
    for i in range(len(labels_depr_diff)):
        var_labels_depr_diff.append(np.nanvar(labels_depr_diff[i]))

    # plot ten best samples
    # get pearson corr for each sample
    pearson_corrs = []
    for i in range(len(full_preds)):
        pearson_corrs.append(pearson_mask(full_preds[i], labels[i]))

    pearson_corrs_ctrl = []
    for i in range(len(ctrl_preds)):
        pearson_corrs_ctrl.append(pearson_mask(ctrl_preds[i], labels_ctrl[i]))

    print("#"*20)

    num_plots = 200

    ######### 
    # plot (num_plots) best samples
    #########
    # best_samples = sorted(range(len(pearson_corrs)), key = lambda sub: pearson_corrs[sub])[-num_plots:]

    # get num_plots samples that have the highest pearson correlation and variance in the depr diff labels
    best_samples = sorted(range(len(pearson_corrs)), key = lambda sub: (pearson_corrs[sub], var_labels_depr_diff[sub]))[-num_plots:]

    # print best pearson corrs
    print("List of best pearson correlations: ", [pearson_corrs[i] for i in best_samples])
    best_pcc = [pearson_corrs[i] for i in best_samples]

    print("#"*20)

    for i in range(num_plots):
        print("BEST: ", i)
        print("Pearson Correlation: ", pearson_corrs[best_samples[i]])
        print("Variance in Depr Diff Labels: ", var_labels_depr_diff[best_samples[i]])
        out_loc = output_loc + "/interpretability_panels/sample_" + str(best_samples[i]) + '_' + str(inverse_condition_values[condition_samples[best_samples[i]]]) + "_best_" + transcripts[best_samples[i]] + "_" + genes[best_samples[i]] + ".png"
        interpretability_plot(model, inputs[best_samples[i]], labels[best_samples[i]], labels_ctrl[best_samples[i]], full_preds[best_samples[i]], out_loc, best_pcc[i], transcripts[best_samples[i]], genes[best_samples[i]], str(inverse_condition_values[condition_samples[best_samples[i]]]))

    # ######### 
    # # plot (num_plots) worst samples
    # #########
    # worst_samples = sorted(range(len(pearson_corrs)), key = lambda sub: pearson_corrs[sub])[:num_plots]
    
    # # print worst pearson corrs
    # print("Worst Pearson Correlations: ", [pearson_corrs[i] for i in worst_samples])
    # worst_pcc = [pearson_corrs[i] for i in worst_samples]

    # print("#"*20)

    # for i in range(num_plots):
    #     print("WORST: ", i)
    #     out_loc = output_loc + "/interpretability_panels/sample_" + str(worst_samples[i]) + '_' + str(inverse_condition_values[condition_samples[worst_samples[i]]]) + "_worst_" + transcripts[worst_samples[i]] + "_" + genes[worst_samples[i]] + ".png"
    #     interpretability_plot(model, inputs[worst_samples[i]], labels[worst_samples[i]], labels_ctrl[worst_samples[i]], full_preds[worst_samples[i]], out_loc, worst_pcc[i], transcripts[worst_samples[i]], genes[worst_samples[i]], str(inverse_condition_values[condition_samples[worst_samples[i]]]))


