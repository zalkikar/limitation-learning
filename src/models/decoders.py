import torch
import torch.nn as nn

OUTPUT_SIZE = 300 ## can be changed for integration with mlp or whatever else


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