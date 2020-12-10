# -*- coding: utf-8 -*-

import gensim
import spacy
from spacy.strings import hash_string
import time


def create_google_news_vectors(path, prune_vectors_on_vocab=True):

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

    print('loading GoogleNewsVectors into blank spacy...')
    start = time.process_time()
    create_google_news_vectors("./dat/vectors/GoogleNews-vectors-negative300.bin.gz")
    print(f'complete {time.process_time() - start}')

    """
    start_nlp = spacy.load('./models/spacy-blank-GoogleNews/')
    #start_nlp = spacy.load('en_core_web_lg')

    #nlp = spacy.load('en_core_web_lg')
    nlp = spacy.load('./models/spacy-blank-GoogleNews/')
    nlp.vocab.vectors = nlp.vocab.vectors.from_disk('./models/spacy-pretrain-GoogleNews/model9.bin')

    all_toks = []
    with open('./dat/processed/formatted_movie_lines.txt', 'r', encoding = 'utf-8') as pf:
        for line in pf:
            line = line.replace("\n", "").replace("</s>","").replace("</d>","")
            toks = line.strip().split(' ') # already processed so split by space
            for tok in toks:
                if tok not in all_toks:
                    all_toks.append(tok)

    count = 0
    all_same_vects = True
    for w in all_toks:
        count += 1
        if (hash_string(w) in nlp.vocab.vectors) and (hash_string(w) in start_nlp.vocab.vectors):
            all_same_vects = all(nlp.vocab.vectors[hash_string(w)] == start_nlp.vocab.vectors[hash_string(w)])
            print(count, w, hash_string(w), all_same_vects)
        else:
            print(count, w, hash_string(w))

        if not all_same_vects:
            break
    """
