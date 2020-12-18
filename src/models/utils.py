import torch
import gensim
import numpy as np

np.random.seed(0) 

def get_model():
    model = gensim.models.Word2Vec.load("../models/custom_w2v_intersect_GoogleNews") # ("./models/custom_w2v")
    EMBED_DIM = model.wv.vectors.shape[1]
    model.wv["<sos>"] = np.random.rand(EMBED_DIM)
    model.wv["<eos>"] = np.random.rand(EMBED_DIM)
    model.init_sims(replace=True) #precomputed l2 normed vectors in-place – saving the extra RAM
    return model

def save_model_instance():
    model = get_model()
    model.save("../models/custom_w2v_intersect_GoogleNews_seq2seqattn")

def get_vectors():
    model = get_model()
    return torch.FloatTensor(model.wv.vectors)

def from_pretrained(embeddings=None, freeze=False):
    if not embeddings:
        embeddings = get_vectors() # 2 D embeddings param
    rows, cols = embeddings.shape
    # A simple lookup table that stores embeddings of a fixed dictionary and size.
    embedding = torch.nn.Embedding(num_embeddings=rows, embedding_dim=cols)
    embedding.weight = torch.nn.Parameter(embeddings)
    # no update if freeze=True (default is False)
    embedding.weight.requires_grad = not freeze
    return embedding
