import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    
    def __init__(self, hidden_size, num_layers,
                 device='cpu', drop_prob=0, lstm=True, feature_norm=False,
                 input_size=300,
                 bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        if lstm:
            memory_cell = nn.LSTM
        else:
            memory_cell = nn.GRU

        self.memory_cell = memory_cell(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       batch_first=True,
                                       # make dropout 0 if num_layers is 1
                                       dropout=drop_prob * (num_layers != 1),
                                       bidirectional=bidirectional)

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



# no embedding layer, assumes embeddings have already been applied?
class EncRnn_pre_embed(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 num_layers, 
                 embed_size=300,
                 device='cpu', 
                 drop_prob=0, 
                 lstm=False, 
                 feature_norm=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        if lstm:
            memory_cell = nn.LSTM
            self.mem_type = 'lstm'
        else:
            memory_cell = nn.GRU
            self.mem_type = 'gru'

        self.memory_cell = memory_cell(input_size = embed_size,
                                       hidden_size = hidden_size,
                                       num_layers = num_layers,
                                       batch_first=True,
                                       # make dropout 0 if num_layers is 1
                                       dropout=drop_prob * (num_layers != 1),
                                       bidirectional=True)

        if feature_norm:
            self.norm = nn.InstanceNorm1d(num_features=embed_size)
        else:
            self.norm = nn.Identity()

        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # transpose to have features as channels
        x = x.transpose(1, 2)
        # run through feature norm
        x = self.norm(x)
        # transpose back
        x = x.transpose(1, 2)

        if self.mem_type == 'lstm':
            out, (final_hidden, cell_hidden) = self.memory_cell(x)
        elif self.mem_type == 'gru':
            out, final_hidden = self.memory_cell(x)

        out = out.transpose(1,0)
        # final hidden has dim (n_layers*n_directions, batch_size, hidden_size)
        # out has dim          (n_tokens, batch_size, hidden_size*n_directions)

        # initial decoder hidden is final hidden state of the forwards and
        # backwards encoder RNNs fed through a linear layer
        concated = torch.cat((final_hidden[-2, :, :], final_hidden[-1, :, :]), dim=1)

        final_hidden = torch.tanh(self.linear(concated))
        
        return out, final_hidden