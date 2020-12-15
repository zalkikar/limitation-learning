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

        self.memory_cell = memory_cell(input_size=hidden_size*2,
                                       hidden_size=hidden_size,
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
        
        self.linear = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
       # print(x.shape)
        out, _ = self.memory_cell(x)
        return self.linear(out)


# no embedding layer, assumes embeddings have already been applied?
class DecRnn_pre_embed(nn.Module):
    def __init__(self, 
                 hidden_size,
                 num_layers,
                 embed_size=300,
                 output_size=300,
                 device='cpu',
                 drop_prob=0,
                 lstm=False):
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

        self.memory_cell = memory_cell(input_size = hidden_size * 2 + embed_size,
                                       hidden_size = hidden_size,
                                       num_layers = num_layers,
                                       batch_first = False,
                                       # make dropout 0 if num_layers is 1
                                       dropout = drop_prob * (num_layers != 1),
                                       bidirectional=False)
        self.Attention = Attention(hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, trg, encoder_outputs, hidden):

        ##print(f'encoder out shape = {encoder_outputs.shape}, encoder hidden shape = {hidden.shape}')

        attention = self.Attention(encoder_outputs, hidden).unsqueeze(1)

        ##print('attention shape = ',attention.shape)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        context = torch.bmm(attention, encoder_outputs).permute(1, 0, 2)
        
        ##print('context shape = ',context.shape)
        ##print('target shape= ',trg.unsqueeze(0).shape)
        
        dec_input = torch.cat((trg.unsqueeze(0), context), dim=2)
        
        ##print('dec input shape = ',dec_input.shape)

        if self.mem_type == 'lstm':
            outputs, (hidden, cell_hidden) = self.memory_cell(dec_input, hidden.unsqueeze(0))
        elif self.mem_type == 'gru':
            outputs, hidden = self.memory_cell(dec_input, hidden.unsqueeze(0))

        prediction = self.linear(outputs.squeeze(0))

        return prediction, hidden.squeeze(0)


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
        ##print(hidden.shape)

        # permute encoder out
        encoder_outputs = encoder_outputs.permute(1, 0 ,2)
        ##print(encoder_outputs.shape)

        # attention gets concatenated of hidden state + enc out
        concat = torch.cat((hidden, encoder_outputs), dim=2)

        # fully connected layer and softmax for attention weight
        energy = torch.tanh(self.l1(concat))
        ##print(energy.shape)

        # attention weight
        attention = self.l2(energy).squeeze(dim=2)
        attention_weight = torch.softmax(attention, dim=1)
        ##print(attention_weight.shape)

        return attention_weight



