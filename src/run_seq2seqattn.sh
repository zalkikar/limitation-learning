#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 150G
#SBATCH --time 0-024:00:00
#SBATCH --job-name run-seq2seqattn
#SBATCH --mail-type=END
#SBATCH --mail-user=rz1567@nyu.edu
#SBATCH --output slurm-%J.log 

module purge

SRCDIR=$SCRATCH/rz1567/deep_rl/limitation-learning/src/

source ~/.bashrc
conda activate irl

python BC.py --epochs 1 --n_hidden 64 --n_layers 2
