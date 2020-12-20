"""
Helper functions and classes used for running GAIL in this conversation based
imitation learning environment. 
"""


import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from models.utils import get_model
from models.config import TOKENS_RAW_CUTOFF
from models.seq2seqattn import init_weights, EncRnn, DecRnn, Seq2SeqAttn
from collections import deque
import random
import torch.optim as optim
import math
#Load in models and helper functions

w2v_model = get_model()
# w2ind from w2v
w2ind = {token: token_index for token_index, token in enumerate(w2v_model.wv.index2word)} 
# sorted vocab words
assert w2v_model.vocabulary.sorted_vocab == True
word_counts = {word: vocab_obj.count for word, vocab_obj in w2v_model.wv.vocab.items()}
word_counts = sorted(word_counts.items(), key=lambda x:-x[1])
words = [t[0] for t in word_counts]
# sentence marker token inds
sos_ind = w2ind['<sos>']
eos_ind = w2ind['<eos>']
# adjusted sequence length
SEQ_LEN = 5 + 2 # sos, eos tokens
# padding token
TRG_PAD_IDX = w2ind["."] # this is 0
# vocab, embed dims
VOCAB_SIZE, EMBED_DIM = w2v_model.wv.vectors.shape

class Discriminator(nn.Module):
    def __init__(self,model,SEQ_LEN):
        super(Discriminator, self).__init__()

        self.state_encoder = model.encoder
        self.action_encoder = model.encoder
        
        self.fc1 = nn.Linear(1280,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,1)
        self.src_len = torch.Tensor([int(SEQ_LEN)])
    def forward(self,x1,x2):
        state_z, _ = self.state_encoder(x1, self.src_len)
        action_z, _ = self.action_encoder(x2, self.src_len)

        state_action = torch.cat([state_z.flatten().unsqueeze(0), action_z.flatten().unsqueeze(0)],dim=1)
        
        state_action = torch.relu(self.fc1(state_action))
        state_action = torch.relu(self.fc2(state_action))
        state_action = torch.sigmoid(self.fc3(state_action))

        return state_action

def get_action_probs(model, input_state, sos_ind, eos_ind, SEQ_LEN, device):
    """
    Given an input sequence and policy, produce a distribution over tokens the predicted token for each step in the sequence. 
    """
    src_tensor = input_state.reshape(1,7).to(device)
    src_len = torch.Tensor([int(SEQ_LEN)])
    encoder_outputs, hidden = model.encoder(src_tensor, src_len)
    #print('encoutshape',encoder_outputs.shape)
    mask = model.create_mask(src_tensor.transpose(1,0)).to(device)
    trg_indexes = [sos_ind]
    attentions = torch.zeros(SEQ_LEN, 1, len(input_state))
    outputs = []
    for i in range(SEQ_LEN):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
        attentions[i] = attention
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
       # print(pred_token)
        if pred_token == eos_ind: # end of sentence.
        
            break
        outputs.append(output)
        
  #  trg_tokens = [words[int(ind)] for ind in trg_indexes]
    #  remove <sos>
    return F.softmax(torch.stack(outputs),dim=2).to(device)



def pad_action(action):
    num_pads = 7 - action.shape[0] - 2
    if num_pads == 0:
        action = torch.cat((torch.LongTensor([sos_ind]), 
                        action.cpu()[1:-1],
                        torch.LongTensor([eos_ind])), 
                       dim=0).to(device)
    if num_pads == 0:
        action =torch.cat((torch.LongTensor([sos_ind]), 
                        action.cpu()[1:-1],torch.LongTensor([0]),
                        torch.LongTensor([eos_ind])), 
                       dim=0).to(device)
        
    if num_pads == 1:
        action = torch.cat((torch.LongTensor([sos_ind]), 
                        action.cpu()[1:-1],torch.LongTensor([0]),
                            torch.LongTensor([0]),
                        torch.LongTensor([eos_ind])), 
                       dim=0).to(device)
    if num_pads == 2:
        action = torch.cat((torch.LongTensor([sos_ind]), 
                        action.cpu()[1:-1],torch.LongTensor([0]),
                            torch.LongTensor([0]),
                            torch.LongTensor([0]),

                        torch.LongTensor([eos_ind])), 
                       dim=0).to(device)
    if num_pads == 3:
        action = torch.cat((torch.LongTensor([sos_ind]), 
                        action.cpu()[1:-1],torch.LongTensor([0]),
                            torch.LongTensor([0]),
                            torch.LongTensor([0]),
                            torch.LongTensor([0]),


                        torch.LongTensor([eos_ind])), 
                       dim=0).to(device)
    if num_pads == 4:
        action = torch.cat((torch.LongTensor([sos_ind]), 
                        action.cpu()[1:-1],torch.LongTensor([0]),
                            torch.LongTensor([0]),
                            torch.LongTensor([0]),
                            torch.LongTensor([0]),
                            torch.LongTensor([0]),


                        torch.LongTensor([eos_ind])), 
                       dim=0).to(device)
    if num_pads == 5:
        action = torch.cat((torch.LongTensor([sos_ind]), 
                        action.cpu()[1:-1],torch.LongTensor([0]),
                            torch.LongTensor([0]),
                            torch.LongTensor([0]),
                            torch.LongTensor([0]),
                            torch.LongTensor([0]),
                            torch.LongTensor([0]),

                        torch.LongTensor([eos_ind])), 
                       dim=0).to(device)
    return action

def get_action(action_probs):
    action = []
    action_log_probs = []
    for i in action_probs:
        m = Categorical(i)
        action_ = m.sample()
        action_log_prob = m.log_prob(action_)# * reward
        action.append(action_.item())
        action_log_probs.append(action_log_prob)
    action = torch.Tensor(action).to(device).long()
    return action, torch.stack(action_log_probs)#.squeeze()

def get_reward(discrim, state, action):
    """
    The reward function according to irl. It's log D(s,a). 
    
    Reward is higher the closer this is to 0, because the more similar it is to an expert action. :
    Is quite close to imitation learning, but hope here is that with such a large number of expert demonstrations and entropy bonuses etc. it learns more than direct imitation. 
    """
    state = state.unsqueeze(0)
    action = action.unsqueeze(0)
    
    with torch.no_grad():
        #TODO: better resize
        return -math.log(discrim(state,action))
    
def train_discrim(memory, discrim_optim, args):
    """
    Training the discriminator. 

    Use binary cross entropy to classify whether 
    or not a sequence was predicted by the expert (real data) or actor. 
    """
    
    criterion = torch.nn.BCELoss() # classify
    learner = torch.stack([memory[i][0] for i in range(len(memory))])
    expert = torch.stack([memory[i][1] for i in range(len(memory))])
    policy_optim.zero_grad()
    discrim_optim.zero_grad()
    # actions came from expert or learner
    discrim_loss = criterion(learner.squeeze(1), torch.ones((args.batch_size, 1)).to(device)) + \
                    criterion(expert.squeeze(1), torch.zeros((args.batch_size, 1)).to(device))
            # discrim loss: predict agent is all wrong, get as close to 0, and predict expert is 1, getting as close to 1 as possible. 
    discrim_loss.backward()

    for _ in range(args.discrim_update_num):
        discrim_optim.step()
            # take these steps, do it however many times specified. 
        #return discrim(states,expert_actions) , discrim(states,actions)
    expert_acc = ((expert < 0.5).float()).mean().item()
    learner_acc = ((learner > 0.5).float()).mean().item()


    return discrim_loss.item(), expert_acc, learner_acc # accuracy, it's the same kind, but because imbalanced better to look at separately. 

def train_policy(memory, policy_optim, args):
    """
    Take several Policy Gradient steps to imporve the policy
    
    using the single step returns to optimize objective. 
    
    
    
    """
    
    rlog_probs = torch.cat([memory[i][0] for i in range(len(memory))])
    policy_optim.zero_grad()
    discrim_optim.zero_grad()

    policy_loss = rlog_probs.sum()
    policy_loss.backward()
    
    for _ in range(args.actor_critic_update_num):

        policy_optim.step()
    return policy_loss.item()