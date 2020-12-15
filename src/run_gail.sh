#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 150G
#SBATCH --time 0-024:00:00
#SBATCH --job-name run-gail
#SBATCH --output slurm-%J.log 

python main.py --logdir 'logs/testlog' --discrim_update_num 10 --actor_critic_update_num 2 
