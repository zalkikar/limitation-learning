#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 150G
#SBATCH --time 0-024:00:00
#SBATCH --job-name run-gail
#SBATCH --output slurm-%J.log 

python main.py --logdir 'logs/v5' --actor_critic_update_num 100 --discrim_update_num 50 --batch_size 32 --hidden_size 8 --num_layers 4
