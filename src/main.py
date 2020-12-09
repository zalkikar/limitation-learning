"""
Main.py file for GAIL implementation on dialog datasets.

Uses command line arguments to maximize flexibility, and run many options in parallel

"""
import os
import gym
import pickle
import argparse
import numpy as np
from collections import deque

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter 

from utils.zfilter import ZFilter
#from model import Actor, Critic, Discriminator
from models.actor import Actor
from models.critic import Critic
from models.discriminator import Discriminator
from train_model import * # THIS IS MISSING RIGHT NOW I THINK
from dialog_environment import DialogEnvironment

device='cpu' # for now


parser = argparse.ArgumentParser(description='PyTorch GAIL for Dialog')

parser.add_argument('--load_model', 
                    type=str, default=None, 
                    help='path to load the saved model')

parser.add_argument('--render', 
                    action="store_true", default=False, 
                    help='if you dont want to render, set this to False')

parser.add_argument('--gamma', 
                    type=float, default=0.99, 
                    help='discounted factor (default: 0.99)')

parser.add_argument('--lamda', 
                    type=float, default=0.98, 
                    help='GAE hyper-parameter (default: 0.98)')

parser.add_argument('--hidden_size', 
                    type=int, default=100,  #TODO
                    help='hidden unit size of actor, critic and discrim networks (default: 100)')

parser.add_argument('--learning_rate', 
                    type=float, default=3e-4, 
                    help='learning rate of models (default: 3e-4)')

parser.add_argument('--l2_rate', 
                    type=float, default=1e-3, 
                    help='l2 regularizer coefficient (default: 1e-3)')

parser.add_argument('--clip_param', 
                    type=float, default=0.2, 
                    help='clipping parameter for PPO (default: 0.2)')

parser.add_argument('--discrim_update_num', 
                    type=int, default=2, 
                    help='update number of discriminator (default: 2)')

parser.add_argument('--actor_critic_update_num', 
                    type=int, default=10, 
                    help='update number of actor-critic (default: 10)')

parser.add_argument('--total_sample_size', 
                    type=int, default=2048, 
                    help='total sample size to collect before PPO update (default: 2048)')

parser.add_argument('--batch_size', 
                    type=int, default=128, 
                    help='batch size to update (default: 128)')

parser.add_argument('--suspend_accu_exp', 
                    type=float, default=0.8,
                    help='accuracy for suspending discriminator about expert data (default: 0.8)')

parser.add_argument('--suspend_accu_gen', 
                    type=float, default=0.8,
                    help='accuracy for suspending discriminator about generated data (default: 0.8)')

parser.add_argument('--max_iter_num', 
                    type=int, default=4000,
                    help='maximal number of main iterations (default: 4000)')

parser.add_argument('--seed', 
                    type=int, default=500,
                    help='random seed (default: 500)')

parser.add_argument('--logdir', 
                    type=str, default='logs/EXPERIMENTNAME',
                    help='tensorboardx logs directory (default: logs/EXPERIMENTNAME')

args = parser.parse_args()



def main():
    env = DialogEnvironment()

    torch.manual_seed(args.seed)

    #TODO
    actor = Actor(hidden_size=3,num_layers=3,device='cuda')
    critic = Critic(hidden_size=1,num_layers=3,device='cuda')
    discrim = Discriminator(input_size = 300, hidden_size=1,device='cuda',num_layers=3)
    
    actor.to(device), critic.to(device), discrim.to(device)
    

    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_optim = optim.Adam(critic.parameters(), lr=args.learning_rate, 
                              weight_decay=args.l2_rate) 
    discrim_optim = optim.Adam(discrim.parameters(), lr=args.learning_rate)

    # load demonstrations

    writer = SummaryWriter(args.logdir)

    if args.load_model is not None: #TODO
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])
        discrim.load_state_dict(ckpt['discrim'])


    
    episodes = 0
    train_discrim_flag = True

    for iter in range(args.max_iter_num):
        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        scores = []

        while steps < args.total_sample_size: 
            state, expert_action, raw_state, raw_expert_action = env.reset()
            score = 0

            
            for _ in range(10000): 

                steps += 1

                mu, std = actor(state.resize(1,30,300)) #TODO: gotta be a better way to resize. 
                action = get_action(mu.cpu(), std.cpu())[0]
                raw_action = get_closest_tokens(action) #TODO
                done= env.step(action)
                irl_reward = get_reward(discrim, state, action)
                if done:
                    mask = 0
                else:
                    mask = 1


                memory.append([state, torch.from_numpy(action).to(device), irl_reward, mask,expert_action])

                score += irl_reward

                if done:
                    break

            
            episodes += 1
            scores.append(score)
        
        score_avg = np.mean(scores)
        print('{}:: {} episode score is {:.2f}'.format(iter, episodes, score_avg))
        writer.add_scalar('log/score', float(score_avg), iter)
        writer.add_scalar('log/expert_acc', float(expert_acc), iter) #logg
        writer.add_scalar('log/learner_acc', float(learner_acc), iter) #logg
        writer.add_scalar('log/avg_acc', float(learner_acc + expert_acc)/2, iter) #logg


        actor.train(), critic.train(), discrim.train()
        if train_discrim_flag:
            expert_acc, learner_acc = train_discrim(discrim, memory, discrim_optim, demonstrations, args) 
            print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
            if expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen:
                train_discrim_flag = False
        train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args)

        if iter % 100:
            score_avg = int(score_avg)
            writer.add_text('log/raw_state', raw_state,iter)
            writer.add_text('log/raw_action', raw_action,iter)
            writer.add_text('log/raw_expert_action', raw_expert_action,iter)


            model_path = os.path.join(os.getcwd(),'save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')

            save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'discrim': discrim.state_dict(),
                'args': args,
                'score': score_avg,
                'accuracy': np.mean([expert_acc,learner_acc])
            }, filename=ckpt_path)

if __name__=="__main__":
    main()


