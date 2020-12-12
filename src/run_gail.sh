#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 150G
#SBATCH --time 0-024:00:00
#SBATCH --job-name run-gail
#SBATCH --output slurm-%J.log 

python main.py --logdir 'logs/v8' --l2_rate 0 --actor_critic_update_num 25 --discrim_update_num 5 --batch_size 64 --hidden_size 16 --num_layers 4
