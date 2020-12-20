import torch
import torch.nn as nn

import numpy as np

from models.seq2seq import Seq2Seq, Seq2SeqAttn_pre_embed

from GAIL import get_action, get_raw_action

def subsample(data, target, n=15):
    return [x[::n] for x in data], [y[::n] for y in target]


class DialogData(torch.utils.data.Dataset):

    def __init__(self, state_vects, subsample_n=None):

        data = []
        targets = []

        for convo_ind, vects in state_vects.items():
            input_state, next_state = vects[0], vects[1]
            # can add raw state here? idk
            data.append(input_state)
            targets.append(next_state)

        assert len(data) == len(targets)

        if subsample_n:
            data, targets = subsample(data, targets, subsample_n)

        self.data = torch.stack(data)
        self.targets = torch.stack(targets)
        
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    d = torch.load('./dat/processed/padded_vectorized_states.pt')
    raw = torch.load('./dat/processed/raw_states.pt')

    for index, vects in d.items():
        # each is N x 300
        input_state, next_state = vects[0], vects[1]
        # raw strings corresponding to embeddings
        raw_input_state, raw_next_state = list(raw.keys())[index], raw[list(raw.keys())[index]]
        print(raw_input_state)
        print(input_state)
        print(raw_next_state)
        print(next_state)
        if index > 1:
            break
    

    model = Seq2Seq(hidden_size=2, num_layers=2)
    print(model)

    for index, vects in d.items():
        # each is N x 300
        input_state, next_state = vects[0], vects[1]
        # raw strings corresponding to embeddings
        raw_input_state, raw_next_state = list(raw.keys())[index], raw[list(raw.keys())[index]]

        #print(input_state.unsqueeze(0).shape)
        mu = model(input_state.unsqueeze(0)).detach()
        #print(mu.shape)
        
        # ACTOR FORMAT
        logstd = torch.zeros_like(mu)
        std = 0.05*torch.exp(logstd)
        action = get_action(mu, std)[0]
        
        print(raw_input_state, get_raw_action(action))

    """
    dataset = DialogData(d)
    print(len(dataset))
    print(dataset[0][0].shape) # initial state at index 0
    print(dataset[0][1].shape) # next state at index 0

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=128,
                                         shuffle=True,
                                         num_workers=0,
                                        )

    #model = Seq2Seq(hidden_size=2, num_layers=2)

    model = Seq2SeqAttn_pre_embed(hidden_size=512, num_layers=1, drop_prob=0.5)
    print(model)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {param_count:,} trainable parameters')

    for index, (data, target) in enumerate(loader):
        
        #print(index, data.shape, target.shape)

        # run through model to test
        result = model(target, data.cpu()).detach()

        print(result.shape)

        break
    """

