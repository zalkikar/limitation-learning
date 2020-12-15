"""
Main.py file for GAIL implementation on dialog datasets.

Uses command line arguments to maximize flexibility, and run many options in parallel

"""

import sys 
sys.path.append('../src')
import os
import pickle
import argparse
import numpy as np
from collections import deque

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter 

from models.actor import Actor
from models.critic import Critic
from models.discriminator import Discriminator
from GAIL import *

from dialog_environment import DialogEnvironment

device='cuda' # for now


parser = argparse.ArgumentParser(description='Limitation Learning')

parser.add_argument('--load_model', 
                    type=str, default=None, 
                    help='path to load the saved model')

parser.add_argument('--gamma', 
                    type=float, default=0.99, 
                    help='discounted factor (default: 0.99)')

parser.add_argument('--lamda', 
                    type=float, default=0.98, 
                    help='GAE hyper-parameter (default: 0.98)')


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
                    type=int, default=4, 
                    help='update number of discriminator (default: 2)')

parser.add_argument('--actor_critic_update_num', 
                    type=int, default=10, 
                    help='update number of actor-critic (default: 10)')

parser.add_argument('--total_sample_size', 
                    type=int, default=4096, 
                    help='total sample size to collect before PPO update (default: 2048)')

parser.add_argument('--batch_size', 
                    type=int, default=128, 
                    help='batch size to update (default: 128)')

parser.add_argument('--suspend_accu_exp', 
                    type=float, default=None,
                    help='accuracy for suspending discriminator about expert data (default: None)')

parser.add_argument('--suspend_accu_gen', 
                    type=float, default=None,
                    help='accuracy for suspending discriminator about generated data (default: None)')

parser.add_argument('--max_iter_num', 
                    type=int, default=4096,
                    help='maximal number of main iterations (default: 4000)')

parser.add_argument('--seed', 
                    type=int, default=42,
                    help='random seed (default: 500)')

parser.add_argument('--logdir', 
                    type=str, default='logs/sunday_v1',
                    help='tensorboardx logs directory (default: logs/EXPERIMENTNAME)')

parser.add_argument('--hidden_size', 
                    type=int, default=128,
                    help='New sequence length of the representation produced by the encoder/decoder RNNs. (default: 1024)')
parser.add_argument('--num_layers', 
                    type=int, default=2,
                    help='Number of layers in the respective RNNs (default: 2)')

parser.add_argument('--seq_len', 
                    type=int, default=5,
                    help='length of input and response sequences (default: 60, which is also max)')
parser.add_argument('--input_size', 
                    type=int, default=50,
                    help='DO NOT CHANGE UNLESS NEW EMBEDDINGS ARE MADE. Dimensionality of embeddings (default: 300)')

args = parser.parse_args()




def main():
    env = DialogEnvironment()
    experiment_name = args.logdir.split('/')[1] #model name

    torch.manual_seed(args.seed)

    #TODO
    actor = Actor(hidden_size=args.hidden_size,num_layers=args.num_layers,device='cuda',input_size=args.input_size,output_size=args.input_size)
    critic = Critic(hidden_size=args.hidden_size,num_layers=args.num_layers,input_size=args.input_size,seq_len=args.seq_len)
    discrim = Discriminator(hidden_size=args.hidden_size,num_layers=args.hidden_size,input_size=args.input_size,seq_len=args.seq_len)
    
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
        similarity_scores = []
        while steps < args.total_sample_size: 
            scores = []
            similarity_scores = []
            state, expert_action, raw_state, raw_expert_action = env.reset()
            score = 0
            similarity_score = 0
            state = state[:args.seq_len,:]
            expert_action = expert_action[:args.seq_len,:]
            state = state.to(device)
            expert_action = expert_action.to(device)
            for _ in range(10000): 

                steps += 1

                mu, std = actor(state.resize(1,args.seq_len,args.input_size)) #TODO: gotta be a better way to resize. 
                action = get_action(mu.cpu(), std.cpu())[0]
                for i in range(5):
                    emb_sum = expert_action[i,:].sum().cpu().item()
                    if emb_sum == 0:
                       # print(i)
                        action[i:,:] = 0 # manual padding
                        break

                done= env.step(action)
                irl_reward = get_reward(discrim, state, action, args)
                if done:
                    mask = 0
                else:
                    mask = 1


                memory.append([state, torch.from_numpy(action).to(device), irl_reward, mask,expert_action])
                score += irl_reward
                similarity_score += get_cosine_sim(expert=expert_action,action=action.squeeze(),seq_len=5)
                #print(get_cosine_sim(s1=expert_action,s2=action.squeeze(),seq_len=5),'sim')
                if done:
                    break

            episodes += 1
            scores.append(score)
            similarity_scores.append(similarity_score)

        score_avg = np.mean(scores)
        similarity_score_avg = np.mean(similarity_scores)
        print('{}:: {} episode score is {:.2f}'.format(iter, episodes, score_avg))
        print('{}:: {} episode similarity score is {:.2f}'.format(iter, episodes, similarity_score_avg))

        actor.train(), critic.train(), discrim.train()
        if train_discrim_flag:
            expert_acc, learner_acc = train_discrim(discrim, memory, discrim_optim, args) 
            print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
            writer.add_scalar('log/expert_acc', float(expert_acc), iter) #logg
            writer.add_scalar('log/learner_acc', float(learner_acc), iter) #logg
            writer.add_scalar('log/avg_acc', float(learner_acc + expert_acc)/2, iter) #logg
            if args.suspend_accu_exp is not None: #only if not None do we check.
                if expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen:
                    train_discrim_flag = False

        train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args)
        writer.add_scalar('log/score', float(score_avg), iter)
        writer.add_scalar('log/similarity_score', float(similarity_score_avg), iter)
        writer.add_text('log/raw_state', raw_state[0],iter)
        raw_action = get_raw_action(action) #TODO
        writer.add_text('log/raw_action', raw_action,iter)
        writer.add_text('log/raw_expert_action', raw_expert_action,iter)

        if iter % 100:
            score_avg = int(score_avg)
            # Open a file with access mode 'a'
            file_object = open(experiment_name'.txt', 'a')

            result_str = str(iter) + '|' + raw_state[0] + '|' + raw_action + '|' + raw_expert_action + '\n'
            # Append at the end of file
            file_object.write(result_str)
            # Close the file
            file_object.close()

            model_path = os.path.join(os.getcwd(),'save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            ckpt_path = os.path.join(model_path, experiment_name + '_ckpt_'+ str(score_avg)+'.pth.tar')

            save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'discrim': discrim.state_dict(),
                'args': args,
                'score': score_avg,
            }, filename=ckpt_path)
if __name__=="__main__":
    main()


