# libraries
from Bio import SeqIO
import pandas as pd

condition = 'CTRL'

# load test file
test_ds = pd.read_csv("../data/dh/test_0.3_NZ_20_PercNan_0.05.csv")

test_ds = test_ds[test_ds["condition"] == condition]

test_genes = list(test_ds["gene"])

total_genes = list(set(test_genes))
print(len(total_genes))

# read fasta file
fasta_file = "/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/reference/ensembl.cds.fa"

gene_id_lis  = []
transcript_id_lis = []
sequences_lis = []

for record in SeqIO.parse(fasta_file, "fasta"):
    rec = record.description.split(' ')[3]
    gene_id = rec.split(':')[1]
    transcript_id = record.description.split(' ')[0]

    if gene_id not in total_genes and len(str(record.seq)) >= 120: # 40 codons min length for the genes
        gene_id_lis.append(gene_id)
        sequences_lis.append(str(record.seq))
        transcript_id_lis.append(transcript_id)

# create dataframe
df = pd.DataFrame({'gene': gene_id_lis, 'sequence': sequences_lis, 'transcript': transcript_id_lis})

# for each gene duplicate, choose the longest sequence
df = df.sort_values('sequence', ascending=False).drop_duplicates('gene').sort_index()

# remove those genes that had a 'N' in the sequence
df = df[~df['sequence'].str.contains('N')]

# for each gene only keep the longest transcript
# go through each gene and keep the longest transcript
for gene in list(set(list(df['gene']))):
    df_gene = df[df['gene'] == gene]
    if len(df_gene) > 1:
        longest_transcript = df_gene[df_gene['sequence'].str.len() == df_gene['sequence'].str.len().max()]
        df = df[~df['gene'].isin([gene])]
        df = pd.concat([df, longest_transcript])

# save dataframe
df.to_csv("data_mouseNonTest/new_mouse_genes_not_in_test_" + condition + ".csv", index=False)

# print number of transcripts
print(len(list(set(list(df['gene'])))))

print(df)