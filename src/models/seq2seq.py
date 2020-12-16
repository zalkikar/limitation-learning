import torch
import torch.nn as nn

import random

from models.encoders import EncoderRNN, EncRnn_pre_embed
from models.decoders import DecoderRNN, DecRnn_pre_embed

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

        self.decoder = DecoderRNN(hidden_size=hidden_size,
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
