#!/bin/bash
#SBATCH --job-name xlnet_darnell # Name for your job
#SBATCH --ntasks 4              # Number of (cpu) tasks
#SBATCH --time  3600         # Runtime in minutes.
#SBATCH --mem 12000             # Reserve x GB RAM for the job
#SBATCH --partition gpu         # Partition to submit
#SBATCH --qos staff             # QOS
#SBATCH --gres gpu:gtx1080:1            # Reserve 1 GPU for usage (titartx, gtx1080)
#SBATCH --chdir /nfs_home/nallapar/riboclette/src/models/xlnet # Directory of the job

# RUN BENCHMARK
source /nfs_home/nallapar/rb-prof/bio_embeds/bin/activate
python model_dh.py
