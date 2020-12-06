import torch
from torch.autograd import Variable
import numpy as np


def np_to_var(d, cuda=True, requires_grad=False):
    d = torch.from_numpy(d).type(torch.FloatTensor).contiguous()
    if cuda:
        d = d.cuda()
    return Variable(d, requires_grad=requires_grad)


def var_to_np(v):
    d = v.data
    if d.is_cuda:
        d = d.cpu()
    return d.numpy()


def pad_sequences(sequences, cuda=True):
    lengths = np.array([s.shape[0] for s in sequences])
    max_length = max(lengths)
    dim = sequences[0].shape[1]
    batch_size = len(sequences)
    data = np.zeros([max_length, batch_size, dim])
    for i, seq in enumerate(sequences):
        seq_length = seq.shape[0]
        data[0:seq_length, i, :] = seq
    data = np_to_var(data, cuda)
    return data, lengths


def unpad_sequences(tensor, lengths):
    n_sequences = len(lengths)
    seq_list = []
    for i in range(n_sequences):
        seq = tensor[i, 0:lengths[i], :]
        seq_list.append(seq)
    return seq_list
