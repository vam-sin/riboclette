# libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr 
from torchmetrics.functional import pearson_corrcoef
import itertools
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.interpolate import make_interp_spline
from captum.attr import IntegratedGradients

# global variables defined for all the function to use
# one-hot encoding for the conditions
condition_values = {'CTRL': 0, 'LEU': 1, 'ARG': 2}
inverse_condition_values = {0: 'CTRL', 1: 'LEU', 2: 'ARG'}

# one-hot encoding for the codons
id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

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

    # convert to lists and remove padding
    preds = preds.tolist()
    preds = [pred[:lengths[i]] for i, pred in enumerate(preds)]

    labels = labels.tolist()
    labels = [label[:lengths[i]] for i, label in enumerate(labels)]

    inputs = inputs.tolist()
    inputs = [input[:lengths[i]] for i, input in enumerate(inputs)]

    # get the condition for each of the samples
    # do / with 64 to get the condition
    condition_samples = []
    for i in range(len(inputs)):
        condition_samples.append(inputs[i][0] // 64)


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

    # for loop which takes the original transcript one-hot sequence and converts into the (64*(condition_one-hot) + x) version (stored in sequences_ds) 
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

    # for loop to get the control label for all the transcripts
    for i in range(len(inputs)):
        condition_sample = inverse_condition_values[condition_samples[i]]
        # search for inputs[i] in sequences_ds get index
        for j in range(len(sequences_ds)):
            if sequences_ds[j] == inputs[i] and condition_sample == condition_list[j]:
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
    for i in range(len(full_preds)):
        pearson_corrs.append(pearson_mask(full_preds[i], labels[i]))

    # pearson mean for each condition
    pearson_means = [[] for i in range(3)]
    for i in range(len(pearson_corrs)):
        pearson_means[condition_samples[i]].append(pearson_corrs[i])
    
    # print means
    for i in range(len(pearson_means)):
        print("Condition: ", inverse_condition_values[i], " Mean: ", np.mean(pearson_means[i]), " Std: ", np.std(pearson_means[i]), " Num Samples: ", len(pearson_means[i]))

    pearson_corrs_ctrl = []
    for i in range(len(ctrl_preds)):
        pearson_corrs_ctrl.append(pearson_mask(ctrl_preds[i], labels_ctrl[i]))

    # output all the predictions into df from lists
    output_analysis_df = pd.DataFrame(list(zip(transcripts, genes, codon_sequences, pearson_corrs, pearson_corrs_ctrl, condition_list)), columns =['Transcript', 'Gene', 'Sequence', 'Full Prediction Pearson Correlation', 'Control Prediction Pearson Correlation', 'Deprivation Condition'])
    output_analysis_df.to_csv(output_loc + "/analysis.csv", index=False)

    num_plots = 10

    # get ten best samples
    best_samples = sorted(range(len(pearson_corrs)), key = lambda sub: pearson_corrs[sub])[-num_plots:]
    # print best pearson corrs
    print("Best Pearson Correlations: ", [pearson_corrs[i] for i in best_samples])

    ### Checking the possibility that the model is predicting the same thing for deprivation and control
    depr_control_truth_pcc_bnw = []
    perf_bnw = []
    # for the 10 best samples 
    # get pcc for the deprivation with the control prediction (if it is not control)
    depr_control_pred_pcc_best = []
    depr_control_truth_pcc_best = []
    for i in range(num_plots):
        if inverse_condition_values[condition_samples[best_samples[i]]] != 'CTRL':
            depr_control_pred_pcc_best.append(pearson_mask(ctrl_preds[best_samples[i]], full_preds[best_samples[i]]))
            depr_control_truth_pcc_best.append(pearson_mask(labels_ctrl[best_samples[i]], labels[best_samples[i]]))
            depr_control_truth_pcc_bnw.append(pearson_mask(labels_ctrl[best_samples[i]], labels[best_samples[i]]))
            perf_bnw.append(pearson_corrs[best_samples[i]])

    # print mean of the pccs
    print("BEST SAMPLES: ", len(depr_control_pred_pcc_best))
    print("Mean of the PCCs between the control prediction and the full prediction: ", np.mean(depr_control_pred_pcc_best))
    print("Mean of the PCCs between the control truth and the full truth: ", np.mean(depr_control_truth_pcc_best))

    for i in range(num_plots):
        out_loc = output_loc + "/full_plots/sample_" + str(best_samples[i]) + '_' + str(inverse_condition_values[condition_samples[best_samples[i]]]) + "_best_" + transcripts[best_samples[i]] + "_" + genes[best_samples[i]] + ".png"
        pearson_corr_full = pearson_mask(full_preds[best_samples[i]], labels[best_samples[i]])
        pearson_corr_ctrl = pearson_mask(ctrl_preds[best_samples[i]], labels_ctrl[best_samples[i]])

        print("Transcript: ", transcripts[best_samples[i]], " Gene: ", genes[best_samples[i]], " Condition: ", inverse_condition_values[condition_samples[best_samples[i]]], " Pearson Correlation FULL: ", pearson_corr_full, " Pearson Correlation CTRL: ", pearson_corr_ctrl)
        min_y = min(min(full_preds[best_samples[i]]), min(depr_diffs[best_samples[i]]), min(ctrl_preds[best_samples[i]])) - 0.1
        max_y = max(max(full_preds[best_samples[i]]), max(depr_diffs[best_samples[i]]), max(ctrl_preds[best_samples[i]])) + 0.1

        min_diff_y = min(min(depr_diffs[best_samples[i]]), min(labels_depr_diff[best_samples[i]])) - 0.1
        max_diff_y = max(max(depr_diffs[best_samples[i]]), max(labels_depr_diff[best_samples[i]])) + 0.1

        # subplots for ctrl, depr, full, labels
        fig, axs = plt.subplots(7, 1, figsize=(20, 10))
        axs[0].set_title("Pearson Correlation FULL: " + str(pearson_corr_full) + " Pearson Correlation CTRL: " + str(pearson_corr_ctrl) + " Condition: " + str(inverse_condition_values[condition_samples[best_samples[i]]]))
        # remove axes for axs[0]
        axs[0].axis('off')
        axs[1].plot(ctrl_preds[best_samples[i]], color='#2ecc71')
        # axs[1].set_title("CTRL PRED")
        # set limit to the max and min of full preds
        axs[1].set_ylim([min_y, max_y])

        axs[2].plot(depr_diffs[best_samples[i]], color='#e74c3c')
        # axs[2].set_title("DEPR DIFF PRED")
        axs[2].set_ylim([min_diff_y, max_diff_y])

        # label depr diff
        axs[3].plot(labels_depr_diff[best_samples[i]], color='#e74c3c')
        # axs[3].set_title("LABEL DEPR DIFF")
        axs[3].set_ylim([min_diff_y, max_diff_y])

        axs[4].plot(full_preds[best_samples[i]], color='#3498db')
        axs[4].set_ylim([min_y, max_y])
        # axs[4].set_title("FULL PRED")

        axs[5].plot(labels[best_samples[i]], color='#f39c12')
        # axs[4].set_title("LABEL FULL")

        axs[6].plot(labels_ctrl[best_samples[i]], color='#f39c12')
        # axs[5].set_title("LABEL CTRL")

        fig.tight_layout()

        plt.savefig(out_loc)
        plt.clf()

    # plot ten worst samples
    worst_samples = sorted(range(len(pearson_corrs)), key = lambda sub: pearson_corrs[sub])[:num_plots]
    # print worst pearson corrs
    print("Worst Pearson Correlations: ", [pearson_corrs[i] for i in worst_samples])
    # get idx of the worst samples 

    ### Checking the possibility that the model is predicting the same thing for deprivation and control
    # for the 10 worst samples 
    # get pcc for the deprivation with the control prediction (if it is not control)
    depr_control_pred_pcc_worst = []
    depr_control_truth_pcc_worst = []
    for i in range(num_plots):
        if inverse_condition_values[condition_samples[worst_samples[i]]] != 'CTRL':
            depr_control_pred_pcc_worst.append(pearson_mask(ctrl_preds[worst_samples[i]], full_preds[worst_samples[i]]))
            depr_control_truth_pcc_worst.append(pearson_mask(labels_ctrl[worst_samples[i]], labels[worst_samples[i]]))

    # print mean of the pccs
    print("WORST SAMPLES: ", len(depr_control_pred_pcc_worst))
    print("Mean of the PCCs between the control prediction and the full prediction: ", np.mean(depr_control_pred_pcc_worst))
    print("Mean of the PCCs between the control truth and the full truth: ", np.mean(depr_control_truth_pcc_worst))

    for i in range(num_plots):
        out_loc = output_loc + "/full_plots/sample_" + str(worst_samples[i]) + '_' + str(inverse_condition_values[condition_samples[worst_samples[i]]]) + "_worst_" + transcripts[worst_samples[i]] + "_" + genes[worst_samples[i]] + ".png"
        pearson_corr_full = pearson_mask(full_preds[worst_samples[i]], labels[worst_samples[i]])
        pearson_corr_ctrl = pearson_mask(ctrl_preds[worst_samples[i]], labels_ctrl[worst_samples[i]])
        min_y = min(min(full_preds[worst_samples[i]]), min(depr_diffs[worst_samples[i]]), min(ctrl_preds[worst_samples[i]])) - 0.1
        max_y = max(max(full_preds[worst_samples[i]]), max(depr_diffs[worst_samples[i]]), max(ctrl_preds[worst_samples[i]])) + 0.1

        min_diff_y = min(min(depr_diffs[worst_samples[i]]), min(labels_depr_diff[worst_samples[i]])) - 0.1
        max_diff_y = max(max(depr_diffs[worst_samples[i]]), max(labels_depr_diff[worst_samples[i]])) + 0.1

        print("Transcript: ", transcripts[worst_samples[i]], " Gene: ", genes[worst_samples[i]], " Condition: ", inverse_condition_values[condition_samples[worst_samples[i]]], " Pearson Correlation FULL: ", pearson_corr_full, " Pearson Correlation CTRL: ", pearson_corr_ctrl)
        # subplots for ctrl, depr, full, labels
        fig, axs = plt.subplots(7, 1, figsize=(20, 10))
        axs[0].set_title("Pearson Correlation FULL: " + str(pearson_corr_full) + " Pearson Correlation CTRL: " + str(pearson_corr_ctrl) + " Condition: " + str(inverse_condition_values[condition_samples[worst_samples[i]]]))
        # remove axes for axs[0]
        axs[0].axis('off')
        axs[1].plot(ctrl_preds[worst_samples[i]], color='#2ecc71')
        # axs[1].set_title("CTRL")
        # set y lims to full preds
        axs[1].set_ylim([min_y, max_y])

        axs[2].plot(depr_diffs[worst_samples[i]], color='#e74c3c')
        # axs[2].set_title("DEPR DIFF")
        axs[2].set_ylim([min_diff_y, max_diff_y])

        # label depr diff
        axs[3].plot(labels_depr_diff[worst_samples[i]], color='#e74c3c')
        # axs[3].set_title("LABEL DEPR DIFF")
        axs[3].set_ylim([min_diff_y, max_diff_y])

        axs[4].plot(full_preds[worst_samples[i]], color='#3498db')
        # axs[4].set_title("FULL PRED")
        axs[4].set_ylim([min_y, max_y])

        axs[5].plot(labels[worst_samples[i]], color='#f39c12')
        # axs[4].set_title("LABEL FULL")

        axs[6].plot(labels_ctrl[worst_samples[i]], color='#f39c12')
        # axs[5].set_title("LABEL CTRL")

        fig.tight_layout()

        # set title to pearson corr and condition

        plt.savefig(out_loc)
        plt.clf()

    ### Checking the possibility that the model is predicting the same thing for deprivation and control
    # for the 10 worst samples 
    # get pcc for the deprivation with the control prediction (if it is not control)
    depr_control_pred_pcc_total = []
    depr_control_truth_pcc_total = []
    perf_total = []
    for i in range(len(condition_samples)):
        # print(inverse_condition_values[condition_samples[i]])
        if inverse_condition_values[condition_samples[i]] != 'CTRL':
            depr_control_pred_pcc_total.append(pearson_mask(ctrl_preds[i], full_preds[i]))
            depr_control_truth_pcc_total.append(pearson_mask(labels_ctrl[i], labels[i]))
            perf_total.append(pearson_corrs[i])

    # print mean of the pccs
    print("TOTAL SAMPLES: ", len(depr_control_pred_pcc_total))
    print("Mean of the PCCs between the control prediction and the full prediction: ", np.mean(depr_control_pred_pcc_total))
    print("Mean of the PCCs between the control truth and the full truth: ", np.mean(depr_control_truth_pcc_total))

    print("PCC bn C-Full PCC and Performance: ", spearmanr(depr_control_pred_pcc_total, perf_total))

def quantile_metric(preds, labels, inputs, output_loc):
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
    # do a / with 64 to get the condition
    for i in range(len(inputs)):
        condition_samples.append(inputs[i][0] // 64)

    genes = []
    transcripts = []

    ds = pd.read_csv('/nfs_home/nallapar/riboclette/src/models/xlnet/data/dh/ribo_test_ALL-NA_dh_0.5_NZ_20_PercNan_0.05.csv')

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
        pearson_corrs.append(pearson_mask(full_preds[i], labels[i], mask[i], lengths[i]))

    # pearson mean for each condition
    pearson_means = [[] for i in range(3)]
    for i in range(len(pearson_corrs)):
        pearson_means[condition_samples[i]].append(pearson_corrs[i])
    
    # print means for each condition
    for i in range(len(pearson_means)):
        print("Condition: ", inverse_condition_values[i], " Mean: ", np.mean(pearson_means[i]), " Std: ", np.std(pearson_means[i]), " Num Samples: ", len(pearson_means[i]))

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
            f1_val = f1_score_masked(pred_quantile, label_quantile, mask[i], lengths[i])
            prec_val = prec_score_masked(pred_quantile, label_quantile, mask[i], lengths[i])
            recall_val = recall_score_masked(pred_quantile, label_quantile, mask[i], lengths[i])

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
    plt.clf()

    # using quantile means get the area under the curve
    auc_f1 = np.trapz(quantile_means_f1, dx=0.1)
    auc_prec = np.trapz(quantile_means_prec, dx=0.1)
    auc_recall = np.trapz(quantile_means_recall, dx=0.1)
    print("AUC F1 Score: ", auc_f1)
    print("AUC Precision Score: ", auc_prec)
    print("AUC Recall Score: ", auc_recall)

def captum_interpretability(model, inputs, labels):
    model.eval()

    ig = IntegratedGradients(model)

    # make mask removing those that have a input of -100
    mask = inputs != -100.0

    # get lengths of each sequence
    lengths = np.sum(mask, axis=1)

    # convert to lists and remove padding
    labels = labels.tolist()
    inputs = inputs.tolist()

    labels = [label[:lengths[i]] for i, label in enumerate(labels)]
    inputs = [input[:lengths[i]] for i, input in enumerate(inputs)]

    # convert back to tensors
    label_sample = torch.tensor(labels[0])
    input_sample = torch.tensor(inputs[0])

    # add dimension for batch size
    label_sample = label_sample.view(1, -1)
    input_sample = input_sample.view(1, -1)

    # input change to long
    input_sample = input_sample.long()

    # print dtypes and shapes
    print(label_sample.dtype, input_sample.dtype)
    

    print(label_sample.shape, input_sample.shape)

    # get attributions
    attributions, deltas = ig.attribute(input_sample, target=label_sample, return_convergence_delta=True)
    print(attributions.shape, deltas.shape)
    # get the mean attributions for each token
    mean_attributions = torch.mean(attributions, dim=0)

    print(mean_attributions)

