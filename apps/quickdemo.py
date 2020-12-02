import torch
import pickle

d = torch.load('./dat/preprocess/padded_vectorized_states.pt')#, map_location=lambda storage, loc: storage.cuda(1))
raw = torch.load('./dat/preprocess/raw_states.pt')#, map_location=lambda storage, loc: storage.cuda(1))

for index, vects in d.items():
    input_state, next_state = vects[0], vects[1]
    raw_input_state, raw_next_state = list(raw.keys())[index], raw[list(raw.keys())[index]]
    print(raw_input_state,"\n", vects[0])
    print(raw_next_state,"\n", vects[1])

    if index > 1:
        break


