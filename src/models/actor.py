import torch
import torch.nn as nn

from models.seq2seq import Seq2Seq, Seq2SeqAttn_pre_embed


class Actor(nn.Module):
    """
    Direct application of Sequence to Sequence Network. Input a state and receive a reply. 
    
    """
    def __init__(self,  hidden_size, num_layers,
                 device='cuda', drop_prob=0, lstm=True, feature_norm=False,
                 input_size=100,output_size=100,
                 bidirectional=True):
        super().__init__()
        
        self.seq2seq = Seq2Seq(hidden_size=hidden_size, num_layers=num_layers,
                 device='cuda', drop_prob=drop_prob, lstm=lstm, feature_norm=feature_norm,
                 input_size=input_size,
                output_size=output_size,
                 bidirectional=bidirectional)
    
    def forward(self,x):
        
        mu = self.seq2seq(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, 0.05*std # output is standard deviation 1 and mean value for gaussian distribution at each point in embedding.

        #unit norm, 
