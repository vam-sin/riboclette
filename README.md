# 🧬🧑🏾‍💻 Riboclette: Conditional Deep Learning Model Reveals Translation Elongation Determinants during Amino Acid Deprivation

Welcome to **Riboclette**, a transformer-based deep learning model for predicting ribosome densities under various nutrient-deprivation conditions. Follow this tutorial to get started! 🚀

---

## Pip Package

Riboclette can be easily installed as a package using which you can make predictions on new gene sequences, and obtain model derived attributions to understand the predictions! 

🧀 **Package Documentation**: [Riboclette on PyPI](https://pypi.org/project/riboclette/)

```bash
pip install riboclette
```

---

## Web Server 🌐🧬

We provide a web-based server where you can explore codon-level attributions for different genes in the dataset. This server allows you to visualize and analyze the model's predictions and interpretability results interactively.

🔗 **Server Link**: [Ribotly](https://lts2.epfl.ch/ribotly/)

On the server, you can:
- Select genes of interest from the dataset.
- View codon-level attributions for each gene.
- Analyze how nutrient-deprivation conditions affect ribosome densities at a single codon resolution.

---

## Code Tutorial 📖✨

### 2️⃣ Download Data and Checkpoints 📂🔗

Download the processed data and the pre-trained model checkpoints from the following link:

[Download Data and Checkpoints](#)

After downloading:
- Place the data in the `riboclette/data/` folder. 📁
- Place the checkpoints in the `riboclette/checkpoints/` folder. ✅

---

### 3️⃣ Prepare the Dataset 🐁📊

To run the data pre-processing pipeline, run the following command:

```bash
cd /riboclette/preprocessing
papermill processing.ipynb
```

---

### 4️⃣ Train the Riboclette Model 🧠💻

Train the Riboclette model using the following command:

```bash
cd /riboclette/models/xlnet/dh
papermill train.ipynb
```

---

### 5️⃣ Perform Pseudolabeling ➕

#### Train 5 Seed Models 🌱🌱🌱🌱🌱

To perform pseudolabeling, first train 5 seed models of Riboclette:

```bash
cd /riboclette/models/xlnet/dh
papermill train.ipynb -p seed_val {1, 2, 3, 4, 42}
```

#### Generate the Pseudolabeling Dataset 🧬📋

Once all seed models are trained, generate the pseudolabeling dataset:

```bash
cd /riboclette/preprocessing
papermill plabeling.ipynb
```

#### Train Pseudolabeling-Based Models 🧠🔄

Train pseudolabeling-based model using the following command:

```bash
cd /riboclette/models/xlnet/plabel
papermill train.ipynb 
```

---

### 6️⃣ Generate Interpretability Results 🔍🧬

Generate codon-level interpretations for all sequences for the testing set:

```bash
cd /riboclette/models/xlnet/plabel
papermill LIGInterpret.ipynb
```

---

### 7️⃣ Generate Interpretability Results 🔍🧬

Generate motifs derived from random windows chosen from the full dataset:

```bash
cd /riboclette/models/xlnet/plabel
papermill beamSearch.ipynb
```

---

### 8️⃣ Downstream Analysis and Figure Recreation 📈🖼️

Recreate the figures from the Riboclette paper using the downstream analysis scripts provided in the repository. These scripts allow you to analyze the model outputs and generate the figures mentioned in the paper.

#### Steps for Downstream Analysis:

1. Navigate to the downstream analysis folder:
   ```bash
   cd /riboclette/downstream_analysis
   ```

2. Run the analysis notebooks to generate the respective figures:
   ```bash
   papermill figure{2,3,4,5}.ipynb
   ```

3. The generated figures will be saved in the `riboclette/data/results/figures/` folder. 🖼️

---

🎉 **You're all set!** Follow these steps to fully utilize Riboclette for ribosome density prediction, interpretability, and downstream analysis. 🚀