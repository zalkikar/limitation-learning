#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=dialog_states
#SBATCH --mail-type=END
#SBATCH --mail-user=rz1567@nyu.edu
#SBATCH --output=dialog_states_results%j.out
  
# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

# Move into the directory that contains our code
SRCDIR=$SCRATCH/rz1567/deep_rl/limitation-learning/apps

# Activate the conda environment
source ~/.bashrc
conda activate irl

# Execute the script
python run_dialog_process.py --vectors_path ./dat/GoogleNews-vectors-negative300.bin.gz

# And we're done!