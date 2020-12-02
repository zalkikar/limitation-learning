# -*- coding: utf-8 -*-

import gensim
import spacy
from spacy.strings import hash_string


def create_google_news_vectors(path):
    # Load google news vecs in gensim
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    # Init blank english spacy nlp object
    nlp = spacy.blank('en')
    # Set the vectors for our nlp object to the google news vectors
    nlp.vocab.vectors = spacy.vocab.Vectors(
        data=model.vectors, keys=model.index2word)
    print(nlp.vocab.vectors.shape)
    nlp.vocab.vectors.name = 'GoogleNews'
    nlp.to_disk('./models/spacy-blank-GoogleNews/')


if __name__ == "__main__":

    create_google_news_vectors(
        'C:/Users/rayzc/Downloads/GoogleNews-vectors-negative300.bin.gz')

    #nlp = spacy.load('./models/spacy-blank-GoogleNews/')
    # print(nlp.vocab.strings["<person>"])

    """ # example print
    count = 0
    for w in nlp.vocab.strings:
        count+=1
        if count >10:
            break
        print(w,hash_string(w),nlp.vocab.strings[w],
            nlp.vocab.strings[hash_string(w)])
    """
