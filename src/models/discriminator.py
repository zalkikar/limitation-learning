import torch
import torch.nn as nn

from models.encoders import EncoderRNN
from models.decoders import DecoderRNN


class Discriminator(nn.Module):
    """
    Combination of two encoders for the state and action embeddings to predict value. 
    

    Based on the GAIL for question answering paper you shared. 
    """
    #TODO
    def __init__(self, hidden_size, num_layers,
                 device='cpu', drop_prob=0, lstm=True, feature_norm=False,
                 input_size=300):
        
        super().__init__()
        
        self.state_encoder = EncoderRNN(
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=device,
            drop_prob=drop_prob,
            lstm=lstm,
            feature_norm=feature_norm,
            input_size=input_size,
                            )
        self.action_encoder =  EncoderRNN(
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=device,
            drop_prob=drop_prob,
            lstm=lstm,
            feature_norm=feature_norm,
            input_size=input_size,
                            )
        
        self.MLP = nn.Linear(in_features=120,out_features=1) #should we extends
        
        
    def forward(self,state,action):
        state = self.state_encoder(state) # idk how good these are 
        action = self.action_encoder(action)
        # reshape 
        state_action = torch.cat([state,action],dim=2).reshape(-1,120) # idk if this the right way to encoding
        prob = torch.sigmoid(self.MLP(state_action))
        return prob
