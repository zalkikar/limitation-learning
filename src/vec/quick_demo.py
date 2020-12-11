import torch
import gensim

model = gensim.models.Word2Vec.load("./models/custom_w2v_intersect_GoogleNews")
model.init_sims(replace=True) #precomputed l2 normed vectors in-place – saving the extra RAM
print(model.vocabulary.sorted_vocab) # should be True
print(model.wv.vectors.shape)
print(list(model.wv.vocab.keys()))
print(model_vects.shape)


"""
d = torch.load('./dat/processed/padded_vectorized_states.pt') #, map_location=lambda storage, loc: storage.cuda(1))
d2 = torch.load('./dat/processed/vectorized_states.pt')
raw = torch.load('./dat/processed/raw_states.pt') #, map_location=lambda storage, loc: storage.cuda(1))

for index, vects in d2.items():
    input_state, next_state = vects[0], vects[1]
    raw_input_state, raw_next_state = list(raw.keys())[index], raw[list(raw.keys())[index]]
    print(raw_input_state,"\n", vects[0])
    print(raw_next_state,"\n", vects[1])

    if index > 1:
        break

for index, vects in d.items():
    input_state, next_state = vects[0], vects[1]
    raw_input_state, raw_next_state = list(raw.keys())[index], raw[list(raw.keys())[index]]
    print(raw_input_state,"\n", vects[0])
    print(raw_next_state,"\n", vects[1])

    if index > 1:
        break
"""
