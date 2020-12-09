import torch
import torch.nn as nn

from models.seq2seq import Seq2Seq

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
        # each is 60 x 300
        input_state, next_state = vects[0], vects[1]
        # raw strings corresponding to embeddings
        raw_input_state, raw_next_state = list(raw.keys())[index], raw[list(raw.keys())[index]]
        print(raw_input_state)
        print(input_state)
        print(raw_next_state)
        print(next_state)
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

