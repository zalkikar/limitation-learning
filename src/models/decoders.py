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

## IN PROGRESS
#class DecRnn()


class Attention(nn.Module):
    
    def __init__(self, hidden_size, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size

        self.l1 = nn.Linear(hidden_size * 3, hidden_size)
        self.l2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, hidden):
        seq_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]

        # repeat hidden state seq_len times
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # reshape encoder out
        encoder_outputs = encoder_outputs.reshape(1, 0 ,2)
        # attention gets concatenated of hidden state + enc out
        concat = torch.cat((hidden, encoder_outputs), dim=2)
        # fully connected layer and softmax for attention weight
        energy = torch.tanh(self.l1(concat))
        # attention weight
        attention = self.l2(energy).squeeze(dim=2)
        attention_weight = torch.softmax(attention, dim=1)
        return attention_weight




