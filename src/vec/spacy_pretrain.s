#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=spacy_pretrain
#SBATCH --mail-type=END
#SBATCH --mail-user=rz1567@nyu.edu
#SBATCH --output=spacy_tok2vec_finetune_results%j.out
#SBATCH --gres=gpu:1
  
# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

# Move into the directory that contains our code
SRCDIR=$SCRATCH/rz1567/deep_rl/limitation-learning

# Activate the conda environment
source ~/.bashrc
conda activate irl

# Execute the script
python -m spacy pretrain ./dat/processed/formatted_movie_lines.jsonl ./models/spacy-blank-GoogleNews ./models/spacy-pretrain-GoogleNews --n-iter 50
#[--width] [--conv-depth] [--cnn-window] [--cnn-pieces] [--use-chars] [--sa-depth]
#[--embed-rows] [--loss_func] [--dropout] [--batch-size] [--max-length]
#[--min-length]  [--seed] [--n-iter] [--use-vectors] [--n-save-every]
#[--init-tok2vec] [--epoch-start]

# And we're done!