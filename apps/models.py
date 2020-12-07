import torch
import torch.nn as nn

OUTPUT_SIZE = 300 ## can be changed for integration with mlp or whatever else

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


class EncoderRNN(nn.Module):
    
    def __init__(self, hidden_size, num_layers,
                 device='cpu', drop_prob=0, lstm=True, feature_norm=False,
                 input_size=34):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        if lstm:
            memory_cell = nn.LSTM
        else:
            memory_cell = nn.GRU

        self.memory_cell = memory_cell(input_size,
                                       hidden_size,
                                       num_layers,
                                       batch_first=True,
                                       # make dropout 0 if num_layers is 1
                                       dropout=drop_prob * (num_layers != 1),
                                       bidirectional=False)

        if feature_norm:
            self.norm = nn.InstanceNorm1d(num_features=input_size)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        # transpose to have features as channels
        x = x.transpose(1, 2)
        # run through feature norm
        x = self.norm(x)
        # transpose back
        x = x.transpose(1, 2)

        out, _ = self.memory_cell(x)
        return out


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


if __name__ == '__main__':
    d = torch.load('./dat/preprocess/padded_vectorized_states.pt')
    raw = torch.load('./dat/preprocess/raw_states.pt')

    for index, vects in d.items():
        # each is 60 x 300
        input_state, next_state = vects[0], vects[1]
        # raw strings corresponding to embeddings
        raw_input_state, raw_next_state = list(raw.keys())[index], raw[list(raw.keys())[index]]
        if index > 1:
            break

    dataset = DialogData(d)
    print(len(dataset))
    print(dataset[0][0].shape) # initial state at index 0
    print(dataset[0][1].shape) # next state at index 0

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=5,
                                         shuffle=True,
                                         num_workers=0,
                                        )

    model = Seq2Seq(hidden_size=2, num_layers=2)

    for index, (data, target) in enumerate(loader):
        
        print(index, data.shape, target.shape)

        # run through model to test
        result = model(data.cpu()).detach()

        print(result.shape)

        break

