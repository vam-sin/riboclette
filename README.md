# 🧬 Riboclette: Interpretable condition-aware transformer for predicting ribosome densities in nutrient-deprivation conditions

Riboclette is a transformer-based deep learning model that is capable of predicting ribosome densities for genes specifically in the mouse genome and can make these predictions in the Control condition (without any nutrient availability constraints) and also in conditions of specific amino acid deprivations such as Leucine, Isoleucine, Valine, and combinations of these. In this repository we provide links to the raw data and the preprocessing code to prepare the data. It also has the code to train the Riboclette model, along with the baseline models. The predictions of the Riboclette model along with individual codon importance plots. 

## Dataset Preparation

## Riboclette Model Training

### Pseudolabeling

### Interpretability 

## Baseline Models

BiLSTM Model Training - Combined Single-Head Variant

```
cd /riboclette/models/bilstm/csh
papermill bilstm_csh_train.ipynb
```
