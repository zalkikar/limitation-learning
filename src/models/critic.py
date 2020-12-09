import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import EncoderRNN

class Critic(nn.Module):
    """

    Estimate of value function, I need to know what the best architecute


    """
    
    def __init__(self, hidden_size, num_layers,
                 device='cpu', drop_prob=0, lstm=True, feature_norm=False,
                 input_size=30,seq_len=300):
        
        super().__init__()
        # TODO
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.state_encoder = EncoderRNN(
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=device,
            drop_prob=drop_prob,
            lstm=lstm,
            feature_norm=feature_norm,
            input_size=input_size,
                            )
        
        self.fc1 = nn.Linear(in_features=2*self.seq_len*self.hidden_size,out_features=64)
        
        self.fc2 = nn.Linear(in_features=64,out_features=1)
    def forward(self,state):
        state = self.state_encoder(state)
        # reshape 
        state = state.flatten().reshape(state.shape[0],2*self.seq_len*self.hidden_size)
        #return state

        state = F.relu(self.fc1(state)) # idk if this good enough
        state = self.fc2(state)
        
        return state

if __name__ == '__main__':
    print("Initializing a critic network.")
    x = torch.randn(16,10,300)

    model = model = Critic(hidden_size=4,num_layers=4,input_size=300,seq_len=10)

    out = model(x)
    print('inshape =',x.shape)
    print('outshape =',out.shape)
    print('nparams =',sum(p.numel() for p in model.parameters() if p.requires_grad))