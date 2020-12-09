import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders import EncoderRNN


class Critic(nn.Module):
    """

    Estimate of value function, I need to know what the best architecute


    """
    
    def __init__(self, hidden_size, num_layers,
                 device='cpu', drop_prob=0, lstm=True, feature_norm=False,
                 input_size=300):
        
        super().__init__()
        # TODO
        self.state_encoder = EncoderRNN(
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=device,
            drop_prob=drop_prob,
            lstm=lstm,
            feature_norm=feature_norm,
            input_size=input_size,
                            )
        
        self.MLP = nn.Linear(in_features=60,out_features=1)
        
        
    def forward(self,state):
        state = self.state_encoder(state)
        # reshape 
        state = state.reshape(-1,60)
        state = F.relu(self.MLP(state)) # idk if this good enough
        return state