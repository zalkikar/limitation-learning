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
    def __init__(self,hidden_size=1024, num_layers=2,
                 device='cpu', drop_prob=0, lstm=True, feature_norm=False,
                 input_size=30,seq_len=300):
        super().__init__()
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
        self.action_encoder = EncoderRNN(
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=device,
            drop_prob=drop_prob,
            lstm=lstm,
            feature_norm=feature_norm,
            input_size=input_size,
                            )
        
        self.fc1 = nn.Linear(in_features=4*self.seq_len*self.hidden_size,out_features=64)
        self.fc2 = nn.Linear(in_features=64,out_features=1)

        
    def forward(self,state,action):
        state = self.state_encoder(state) # idk how good these are 
        action = self.action_encoder(action)
        # reshape 
       # return state,action
        state_action = torch.cat([state,action],dim=2).flatten().reshape(state.shape[0],4*self.seq_len*self.hidden_size)
       # return state_action
    
        state_action = torch.relu(self.fc1(state_action))
        prob = torch.sigmoid(self.fc2(state_action))
        return prob


if __name__ == '__main__':
    print("Initialized a discrim.")
    x = torch.randn(16,10,300)

    model = Discriminator(hidden_size=1024,num_layers=2,input_size=300,seq_len=10)


    out = model(x,x)
    print('inshape =',x.shape)
    print('outshape =',out.shape)
    print('nparams =',sum(p.numel() for p in model.parameters() if p.requires_grad))
