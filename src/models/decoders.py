import torch
import torch.nn as nn



class DecoderRNN(nn.Module):
    
    def __init__(self, hidden_size, num_layers,input_size,
                 device='cpu', drop_prob=0, lstm=True, feature_norm=False,output_size=300,bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        if lstm:
            memory_cell = nn.LSTM
        else:
            memory_cell = nn.GRU

        self.memory_cell = memory_cell(input_size=hidden_size,
                                       hidden_size=output_size,
                                       num_layers=num_layers,
                                       batch_first=True,
                                       # make dropout 0 if num_layers is 1
                                       dropout=drop_prob * (num_layers != 1),
                                       bidirectional=bidirectional)

        #deprecated arg
        if feature_norm:
            self.norm = nn.InstanceNorm1d(num_features=input_size)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        out, _ = self.memory_cell(x)
        return out