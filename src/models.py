import torch
import torch.nn as nn
import torch.nn.functional as F

OUTPUT_SIZE = 300 ## can be changed for integration with mlp or whatever else



class EncoderRNN(nn.Module):
    
    def __init__(self, hidden_size, num_layers,
                 device='cpu', drop_prob=0, lstm=True, feature_norm=False,
                 input_size=34):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        if lstm:
            memory_cell = nn.LSTM
        else:
            memory_cell = nn.GRU

        self.memory_cell = memory_cell(input_size,
                                       hidden_size,
                                       num_layers,
                                       batch_first=True,
                                       # make dropout 0 if num_layers is 1
                                       dropout=drop_prob * (num_layers != 1),
                                       bidirectional=False)

        if feature_norm:
            self.norm = nn.InstanceNorm1d(num_features=input_size)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        # transpose to have features as channels
        x = x.transpose(1, 2)
        # run through feature norm
        x = self.norm(x)
        # transpose back
        x = x.transpose(1, 2)

        out, _ = self.memory_cell(x)
        return out


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, num_layers,
                 device='cpu', drop_prob=0, lstm=True, feature_norm=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        if lstm:
            memory_cell = nn.LSTM
        else:
            memory_cell = nn.GRU

        self.memory_cell = memory_cell(hidden_size,
                                       OUTPUT_SIZE,
                                       num_layers,
                                       batch_first=True,
                                       # make dropout 0 if num_layers is 1
                                       dropout=drop_prob * (num_layers != 1),
                                       bidirectional=False)

    def forward(self, x):
        out, _ = self.memory_cell(x)
        return out


class Seq2Seq(nn.Module):
    def __init__(self,  hidden_size, num_layers,
                 device='cpu', drop_prob=0, lstm=True, feature_norm=False,
                 input_size=300):
        super().__init__()
        self.encoder = EncoderRNN(hidden_size,
                                  num_layers,
                                  device,
                                  drop_prob,
                                  lstm,
                                  feature_norm,
                                  input_size=input_size,
                                  )
        self.decoder = DecoderRNN(hidden_size,
                                  num_layers,
                                  device,
                                  drop_prob,
                                  lstm,
                                  feature_norm,
                                  )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
class Actor(nn.Module):
    """
    Direct application of Sequence to Sequence Network. Input a state and reply. 
    
    """
    def __init__(self,  hidden_size, num_layers,
                 device='cuda', drop_prob=0, lstm=True, feature_norm=False,
                 input_size=300):
        super().__init__()
        self.seq2seq = Seq2Seq(hidden_size=hidden_size,num_layers=num_layers,device=device, drop_prob=drop_prob, lstm=lstm, feature_norm=feature_norm,
                          input_size = input_size)
    
    def forward(self,x):
        
        mu = self.seq2seq(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, .01*std # Kinda guessed. 

        #unit norm, 

    
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

