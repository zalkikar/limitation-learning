import torch
import torch.nn as nn

from encoders import EncoderRNN
from decoders import DecoderRNN

class Seq2Seq(nn.Module):
    def __init__(self,  hidden_size, num_layers,
                 device='cpu', drop_prob=0, lstm=True, feature_norm=False,
                 input_size=300,
                 bidirectional=True,output_size=300):
        super().__init__()

        self.encoder = EncoderRNN(hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  device=device,
                                  drop_prob=drop_prob,
                                  lstm=lstm,
                                  feature_norm=feature_norm,
                                  bidirectional=bidirectional,
                                  input_size=input_size,
                                  )

        self.decoder = DecoderRNN(hidden_size=2*hidden_size,
                                  num_layers=num_layers,
                                  device=device,
                                  drop_prob=drop_prob,
                                  lstm=lstm,
                                  feature_norm=feature_norm,
                                  bidirectional=bidirectional,
                                  input_size=input_size,
                                  output_size=output_size
                                  )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Seq2SeqAttn(nn.Module):
    def __init__(self,  hidden_size, num_layers,
                 device='cpu', drop_prob=0, lstm=False, feature_norm=False,
                 input_size=300):
        super().__init__()
        self.encoder = EncRnn(hidden_size,
                              num_layers,
                              device,
                              drop_prob,
                              lstm,
                              feature_norm,
                              input_size=input_size,
                              )
        
    def forward(self, x):
        enc_outs, enc_hidden = self.encoder(x)
        print(enc_outs.shape, enc_hidden.shape)
