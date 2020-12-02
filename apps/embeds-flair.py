import torch
import flair

from flair.embeddings import StackedEmbeddings
from flair.embeddings import FlairEmbeddings

flair_forward  = FlairEmbeddings('news-forward-fast')
flair_backward = FlairEmbeddings('news-backward-fast')

stacked_embeddings = StackedEmbeddings( embeddings = [ 
                                                       flair_forward-fast, 
                                                       flair_backward-fast
                                                      ])