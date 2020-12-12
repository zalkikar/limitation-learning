import gensim
from config import W2V_ITERATIONS, TOKENS_RAW_CUTOFF

"""
intersect_word2vec_format() will let you bring vectors from an external file 
into a model that's already had its own vocabulary initialized (as if by build_vocab()). 
That is, it will only load those vectors for which there are already words in the local vocabulary.
This is a way to auto-prune our vector space from the incredible large Google News Vectors 
when the majority are not related to our dataset.

It will by default lock those loaded vectors against any further adjustment during subsequent training, 
but other words in the pre-existing vocabulary may continue to update. However, we DONT want those vectors locked, so
# we change this behavior by supplying a lockf=1.0 value instead of the default 0.0. Now oov tokens (ex. <person>)
and tokens in Google News Vocab will continue to have their vectors adjusted.

*** This is best considered an experimental function and what, if any, benefits it might offer will depend 
on lots of things specific to your setup.***

Im using ‘distributed memory’ (PV-DM) with dm=1.0 default in doc2vec
"""

def processLine(line):
    """
    these: </u> and <u> were present in raw text, as a quick fix and to avoid a rerun, we handle this later
    by dropping "< u > " and "</u >". also "\x92" was present, iso8859 encoding to utf-8 issues....
    """
    line = line.replace("< u > ","").replace("</u >","").replace("\x92","'")
    line = line.lower()
    return line

my_sentences = [] 
# load processed text from a pretrained spacy model, this is a list of list of tokens
# it would be faster to use data streaming, but for now this should work
# data streaming here: https://rare-technologies.com/data-streaming-in-python-generators-iterators-iterables/

my_sentences = [] 
with open('./dat/processed/formatted_movie_lines.txt', 'r', encoding = 'utf-8') as pf:
        for line in pf:
            line = line.replace("\n", "").replace("</s>","").replace("</d>","")
            line = processLine(line)
            # ignore pairs where raw tokens are above cutoff (assume processed)
            if (len(line.split(' ')) > TOKENS_RAW_CUTOFF):
                continue
            my_sentences.append(line.strip().split())

google_wv = gensim.models.KeyedVectors.load_word2vec_format('./dat/vectors/GoogleNews-vectors-negative300.bin.gz', binary=True)
model = gensim.models.Word2Vec(size=300, 
                               min_count=10,          # token is included if it appears at least 10 time in the vocabulary
                               iter = W2V_ITERATIONS, # number of iterations (epochs)
                               alpha = 0.025,         # initial learning rate
                               min_alpha=0.0001,      # lr with linearly drop during training
                               workers = 3,           # Use these many worker threads to train the model (=faster training with multicore machines).
                               max_final_vocab=500000,# no more than this # of unique words allowed (most frequent will be kept)
                               max_vocab_size = None  # Limits the RAM during vocabulary building; 
                                                      # if there are more unique words than this, then prune the infrequent ones. 
                                                      # Every 10 million word types need about 1GB of RAM. Set to None for no limit.
                               )
model.build_vocab(my_sentences)
training_examples_count = model.corpus_count # save now, lines below will convert to 1
print(training_examples_count)
model.build_vocab([list(google_wv.vocab.keys())], update=True)

# see comment at top for function details
model.intersect_word2vec_format('./dat/vectors/GoogleNews-vectors-negative300.bin.gz',binary=True, lockf=1.0)

model.train(my_sentences,
            total_examples=training_examples_count,
            epochs=model.iter)

print(model.vocabulary.sorted_vocab) # should be True
print(model.wv.vectors.shape)
# print(list(model.wv.vocab.keys()))
# model.wv.vocab.keys() actual vocab tokens

model.save("./models/custom_w2v_intersect_GoogleNews")
