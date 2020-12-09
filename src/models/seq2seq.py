import torch
import torch.nn as nn

from models.encoders import EncoderRNN
from models.decoders import DecoderRNN


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