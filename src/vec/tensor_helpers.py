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
    