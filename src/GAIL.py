from collections import deque  
#code for training 
import torch
import numpy as np

import sys
#sys.path.append('../src')
from models import *
import torch.optim as optim
import math
import torch
from torch.distributions import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    
    # This should be normalized. 
    return action


def get_entropy(mu, std):
    dist = Normal(mu, std)
    entropy = dist.entropy().mean()
    return entropy

def log_prob_density(x, mu, std):
    log_prob_density = -(x - mu).pow(2) / (2 * std.pow(2)) \
                     - 0.5 * math.log(2 * math.pi)
    return log_prob_density.sum(1, keepdim=True)

def get_reward(discrim, state, action):
    """
    The reward function according to irl. It's log D(s,a). 
    
    Reward is higher the closer this is to 0, because the more similar it is to an expert action. :
    Is quite close to imitation learning, but hope here is that with such a large number of expert demonstrations and entropy bonuses etc. it learns more than direct imitation. 
    """

    action = torch.Tensor(action).to(device)# turn state into a tensor if not already

    with torch.no_grad():
        return -math.log(discrim(state.resize(1,60,300),action.resize(1,60,300))[0].item())

def save_checkpoint(state, filename):
    torch.save(state, filename)
    

def train_discrim(discrim, memory, discrim_optim, discrim_update_num, clip_param):
    """
    Training the discriminator. 

    Use binary cross entropy to classify whether 
    or not a sequence was predicted by the expert (real data) or actor. 
    """
    states = torch.stack([memory[i][0] for i in range(len(memory))])
    actions = torch.stack([memory[i][1] for i in range(len(memory))])
    rewards = [memory[i][2] for i in range(len(memory))]

    masks = [memory[i][2] for i in range(len(memory))]
    expert_actions = torch.stack([memory[i][4] for i in range(len(memory))])

    criterion = torch.nn.BCELoss() # classify

    for _ in range(discrim_update_num):

        learner = discrim(states, actions) #pass (s,a) through discriminator

        # TODO
       # demonstrations = torch.Tensor([states, expert_actions]) # pass (s,a) of expert through discriminator
        expert = discrim(states,expert_actions) #discrimator "guesses" whether or not these 
        # actions came from expert or learner
        discrim_loss = criterion(learner, torch.ones((states.shape[0], 1)).to(device)) + \
                        criterion(expert, torch.zeros((states.shape[0], 1)).to(device))
                # discrim loss: predict agent is all wrong, get as close to 0, and predict expert is 1, getting as close to 1 as possible. 
        discrim_optim.zero_grad() # gan loss, it tries to always get it right. 
        discrim_loss.backward()
        discrim_optim.step()
            # take these steps, do it however many times specified. 
        #return discrim(states,expert_actions) , discrim(states,actions)
    expert_acc = ((discrim(states,expert_actions) < 0.5).float()).mean() #how often it realized the fake examples were fake
    learner_acc = ((discrim(states,actions) > 0.5).float()).mean() #how often if predicted expert correctly. 

    return expert_acc, learner_acc # accuracy, it's the same kind, but because imbalanced better to look at separately. 
 