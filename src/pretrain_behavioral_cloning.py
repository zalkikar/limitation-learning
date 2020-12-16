"""
To speed up actor/policy training, use behavioral cloning to prime it's predicted mu 
value 


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


parser = argparse.ArgumentParser(description='Bevavioral Cloning Pretrain')

parser.add_argument('--load_model', 
                    type=str, default=None, 
                    help='path to load the saved model')




parser.add_argument('--learning_rate', 
                    type=float, default=3e-4, 
                    help='learning rate of models (default: 3e-4)')

parser.add_argument('--l2_rate', 
                    type=float, default=1e-3, 
                    help='l2 regularizer coefficient (default: 1e-3)')


parser.add_argument('--batch_size', 
                    type=int, default=256, 
                    help='batch size to update (default: 128)')

parser.add_argument('--max_iter_num', 
                    type=int, default=4096,
                    help='maximal number of main iterations (default: 4000)')

parser.add_argument('--seed', 
                    type=int, default=42,
                    help='random seed (default: 500)')

parser.add_argument('--logdir', 
                    type=str, default='logs/pretrain_v1',
                    help='tensorboardx logs directory (default: logs/EXPERIMENTNAME)')

parser.add_argument('--hidden_size', 
                    type=int, default=32,
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



def main():
    env = DialogEnvironment()
    experiment_name = args.logdir.split('/')[1] #model name

    torch.manual_seed(args.seed)

    #TODO
    actor = Actor(hidden_size=args.hidden_size,num_layers=args.num_layers,device='cuda',input_size=args.input_size,output_size=args.input_size)
    
    actor.to(device)
    
    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate)
    # load demonstrations

    writer = SummaryWriter(args.logdir)

    if args.load_model is not None: #TODO
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
 

    
    episodes = 0


    for iter in range(args.max_iter_num):
        actor.eval()
 

        steps = 0
        scores = []
        states = []
        expert_actions = []
        while steps < args.batch_size: 
            scores = []
            similarity_scores = []
            state, expert_action, raw_state, raw_expert_action = env.reset()

            score = 0
            similarity_score = 0
            state = state[:args.seq_len,:]
            expert_action = expert_action[:args.seq_len,:]
            state = state.to(device)
            expert_action = expert_action.to(device)
            states.append(state)
            expert_actions.append(expert_action)



                similarity_score += get_cosine_sim(expert=expert_action,action=action.squeeze(),seq_len=5)
                #print(get_cosine_sim(s1=expert_action,s2=action.squeeze(),seq_len=5),'sim')
                if done:
                    break

            episodes += 1

            similarity_scores.append(similarity_score)
        states = torch.stack(states)
        actions_pred , _ = actor(states)
        expert_actions = torch.stack(expert_actions)



        similarity_score_avg = np.mean(similarity_scores)
        print('{}:: {} episode similarity score is {:.2f}'.format(iter, episodes, similarity_score_avg))

        actor.train()
        loss = F.mse_loss(actions_pred,expert_action)
        actor_optim.zero_grad()
        actor_optim.step() 
        # and this is basically all we need to do



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
            file_object = open(experiment_name+'.txt', 'a')

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



