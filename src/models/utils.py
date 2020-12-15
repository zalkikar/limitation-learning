import torch
import gensim

def get_vectors():
    model = gensim.models.Word2Vec.load("../models/custom_w2v_intersect_GoogleNews") # ("./models/custom_w2v")
    model.init_sims(replace=True) #precomputed l2 normed vectors in-place – saving the extra RAM
    return torch.FloatTensor(model.wv.vectors)

def from_pretrained(embeddings=None, freeze=True):
    if not embeddings:
        embeddings = get_vectors() # 2 D embeddings param
    rows, cols = embeddings.shape
    # A simple lookup table that stores embeddings of a fixed dictionary and size.
    embedding = torch.nn.Embedding(num_embeddings=rows, embedding_dim=cols)
    embedding.weight = torch.nn.Parameter(embeddings)
    # no update
    embedding.weight.requires_grad = not freeze
    return embedding