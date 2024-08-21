# 🧬 Riboclette: Interpretable condition-aware transformer for predicting ribosome densities in nutrient-deprivation conditions

Riboclette is a transformer-based deep learning model that is capable of predicting ribosome densities for genes specifically in the mouse genome and can make these predictions in the Control condition (without any nutrient availability constraints) and also in conditions of specific amino acid deprivations such as Leucine, Isoleucine, Valine, and combinations of these. In this repository we provide links to the raw data and the preprocessing code to prepare the data. It also has the code to train the Riboclette model, along with the baseline models. The predictions of the Riboclette model along with individual codon importance plots. 

## Dataset Preparation 🐁

Download the raw data files from "link" and run the Ribo-DT pipeline to pre-process them.

```
- RiboDT
```

Convert the pre-processed data into a format the machine learning models can use

```
cd /riboclette/preprocessing
papermill processing.ipynb
```

## Riboclette Model Training 💻

Combined Single-Head Variant

```
cd /riboclette/models/xlnet/csh
papermill xlnet_csh_train.ipynb
```

Double-Head Variant

```
cd /riboclette/models/xlnet/csh
papermill xlnet_csh_train.ipynb
```

### Pseudolabeling ➕

In order to perform the pseudolabeling, you need to first train 5 seed models of Riboclette DH

```
cd /riboclette/models/xlnet/csh
papermill xlnet_csh_train.ipynb -p seed_val 1
papermill xlnet_csh_train.ipynb -p seed_val 2
papermill xlnet_csh_train.ipynb -p seed_val 3
papermill xlnet_csh_train.ipynb -p seed_val 4
papermill xlnet_csh_train.ipynb -p seed_val 42
```

Once all of the seed models are trained, the pseudolabeling dataset can be generated

```
cd /riboclette/models/xlnet/pseudolabeling
papermill plabel_dataset_gen.ipynb
```

After the Pseudolabeling datasets have been generated, the different psuedolabeling based models can be trained. There are two pseudolabeling experiment types ("exp1", and "exp2") which can be specified using the "experiment_type" parameter. 

```
cd /riboclette/models/xlnet/pseudolabeling
papermill xlnet_plabel_train.ipynb -p experiment_type [exp1, exp2]
```

### Interpretability

In order to generate codon-level interpretations for all the sequences in the testing set, run the following commands:

```
cd /riboclette/models/xlnet/pseudolabeling
papermill interpret_plabel.ipynb
```

## Baseline Models 🛸

BiLSTM Model Training - Combined Single-Head Variant

```
cd /riboclette/models/bilstm/csh
papermill bilstm_csh_train.ipynb
```

BiLSTM Model Training - Double-Head Variant

```
cd /riboclette/models/bilstm/dh
papermill bilstm_dh_train.ipynb
```
