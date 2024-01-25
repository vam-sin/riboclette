# DataFrame from FASTA

Make a dataframe from the FASTA file in `/path/to/FASTA`.
```python
import pandas as pd
from Bio import SeqIO

fa_path = /path/to/FASTA
data = []
with open(fa_path, mode="r") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        data.append([record.id, str(record.seq)])

# Create transcripts to sequences mapping
sequences_df = pd.DataFrame(data, columns=["transcript", "sequence"])
```

# CapR
`capr.py`: interface of https://github.com/fukunagatsu/CapR.

## 1. Install CapR
```bash
git clone https://github.com/fukunagatsu/CapR /path/to/CapR
cd /path/to/CapR
make
```

## 2. Run Interface
```python
import os
from dotenv import load_dotenv
from riboclette.rna_sstruct.capr import CapRInterface

os.system('echo "CAPR_PATH=path/to/CapR/CapR" >> .env')
load_dotenv()

# sequences_df is a DataFrame with a transcript and a sequence column
input_df = sequences_df
out_df = CapRInterface().get_categories(input_df)
```

# arnie
`arnie.py`: interface of https://github.com/DasLab/arnie.

## 1. Install arnie
```
pip install git+https://github.com/DasLab/arnie
```
## 2. Install ViennaRNA 
Follow **Quick Start** and **User-dir Installation** in `https://github.com/ViennaRNA/ViennaRNA` and install into `/path/to/ViennaRNA`.

## 3. Run Interface
```python
import os
from dotenv import load_dotenv

# Specify ViennaRNA path in arnie.txt
os.system('echo "vienna_2: /path/to/ViennaRNA/bin" >> path/to/arnie.txt')
# Save /path/to/arnie.txt in envirorment
os.system('echo "ARNIEFILE=path/to/arnie.txt" >> .env')
load_dotenv()

# Load after setting ARNIEFILE
from riboclette.rna_sstruct.arnie import get_nucleotide_pairing_proba, get_maximum_expected_accuracy_seq

# sequences_df is a DataFrame with a sequence column
input_df = sequences_df

# Get pairing probability
df_with_bpps = get_nucleotide_pairing_proba(input_df)

# Get pairing probability and maximum expected sequence
df_with_struct = get_maximum_expected_accuracy_seq(input_df)
```