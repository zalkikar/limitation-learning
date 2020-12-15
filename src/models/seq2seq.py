import torch
import torch.nn as nn

import random
from models.encoders import EncoderRNN, EncRnn_pre_embed
from models.decoders import DecoderRNN, DecRnn_pre_embed
from encoders import EncoderRNN, EncRnn_pre_embed
from decoders import DecoderRNN, DecRnn_pre_embed

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


#------------ ONLY WORKS WITH 1 NUM LAYERS FOR NOW

class Seq2SeqAttn_pre_embed(nn.Module):
    def __init__(self,  hidden_size, num_layers,
                 device='cpu', drop_prob=0, lstm=True, feature_norm=False,
                 input_size=300,
                 bidirectional=True,output_size=300):
        super().__init__()
        self.encoder = EncRnn_pre_embed(hidden_size = hidden_size,
                                        num_layers = 1, # forced to 1. 
                                        embed_size = input_size,
                                        device = device,
                                        drop_prob = drop_prob,
                                        lstm = lstm,
                                        feature_norm = feature_norm
                                        )
        self.decoder = DecRnn_pre_embed(hidden_size = hidden_size,
                                        num_layers = 1,
                                        embed_size = output_size,
                                        output_size = output_size,
                                        device = device,
                                        drop_prob = drop_prob,
                                        lstm = lstm
                                        )
        
    def forward(self, x, targets, teacher_forcing_ratio=0.):
        # encoder_outputs : all hidden states of the input sequence (forward and backward)
        # hidden : final forward and backward hidden states, passed through a linear layer
        enc_outs, enc_hidden = self.encoder(x)

        targets = targets.transpose(1,0)
        seq_len = targets.shape[0]
        batch_size = targets.shape[1]
        trg_vocab_size = targets.shape[2]

        # tensor to store decoder's output
        outputs = torch.zeros(seq_len, batch_size, trg_vocab_size)

        trg = targets[0]
        for i in range(1, seq_len):
            
            prediction, hidden = self.decoder(trg, enc_outs, enc_hidden)
            ##print(f'INDEX {i}, trg {trg.shape}, prediction {prediction.shape}, hidden {hidden.shape}')
            
            outputs[i] = prediction

            if random.random() < teacher_forcing_ratio:
                trg = targets[i]
            else:
                trg = prediction#.argmax(1)

        outputs = outputs.transpose(0,1)
        return outputs
