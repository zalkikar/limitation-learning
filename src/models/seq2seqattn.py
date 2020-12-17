import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import get_model, from_pretrained
from models.config import TOKENS_RAW_CUTOFF

import random


class EncRnn(nn.Module):
    
    def __init__(self, hidden_size, num_layers, embed_size,
                 device='cpu', drop_prob=0, lstm=True, feature_norm=False,
                 bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.embedding = from_pretrained()

        self.memory_cell = torch.nn.GRU(input_size=embed_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True,
                                # make dropout 0 if num_layers is 1
                                dropout=drop_prob * (num_layers != 1),
                                bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, src_len):
        x = self.dropout(self.embedding(x))
        # packing for computation and performance
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, batch_first=True, lengths = src_len)
        out, hidden = self.memory_cell(packed_x)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True) # unpack
        out = out.transpose(1,0)
        # initial decoder hidden is final hidden state of the forwards and
        # backwards encoder RNNs fed through a linear layer
        concated = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = torch.tanh(self.linear(concated))
        return out, hidden



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        
        self.attn = nn.Linear((hidden_size * 2) + hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10) # using masking, we can force the attention to only be over non-padding elements.
        return F.softmax(attention, dim = 1)


class DecRnn(nn.Module):
    def __init__(self, hidden_size, num_layers, embed_size, output_size,
                 device='cpu', drop_prob=0, lstm=True, feature_norm=False,
                 bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.attention = Attention(hidden_size)

        self.embedding = from_pretrained()
        
        self.memory_cell = torch.nn.GRU((hidden_size * 2) + embed_size, hidden_size)
        self.linear = nn.Linear((hidden_size * 3)+embed_size, output_size)
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, input, hidden, encoder_outputs, mask):
             
        ##print(input.shape) #input = [batch size]
        ##print(hidden.shape) #hidden = [batch size, dec hid dim]
        ##print(encoder_outputs.shape) #encoder_outputs = [src len, batch size, enc hid dim * 2]
        ##print(mask.shape) #mask = [batch size, src len]
        
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        ##print(embedded.shape)
        a = self.attention(hidden, encoder_outputs, mask)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        ##print(encoder_outputs.shape)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        ##print(weighted.shape)
        dec_input = torch.cat((embedded, weighted), dim = 2)
        ##print(dec_input.shape, hidden.unsqueeze(0).shape)
        output, hidden = self.memory_cell(dec_input, hidden.unsqueeze(0))

        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.linear(torch.cat((output, weighted, embedded), dim = 1))
        ##print(prediction.shape)
        return prediction, hidden.squeeze(0), a.squeeze(1)


class Seq2SeqAttn(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, vocab_size, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.vocab_size = vocab_size
        
    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0) ### used to be != ???
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        src = src.transpose(1,0)
        trg = trg.transpose(1,0)

        ##print(src.shape) #src = [src len, batch size]
        ##print(src_len.shape) #src_len = [batch size]
        ##print(trg.shape) #trg = [trg len, batch size]
        
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
                    
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.vocab_size
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        ##print(outputs)
        
        src = src.transpose(1,0)
        encoder_outputs, hidden = self.encoder(src, src_len)
        src = src.transpose(1,0)
        ##print(encoder_outputs, hidden)

        input = trg[0,:]
        ##print(input)
        
        mask = self.create_mask(src)
        #print(f'src = {src}')
        #print(f'mask = {mask}')

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            #print(input, top1)
            input = trg[t] if teacher_force else top1
            
        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
