import spacy
from spacy_helpers import add_pipes_from_pretrained
from matplotlib import pyplot as plt

lines = []
with open('./dat/preprocess/formatted_movie_lines.txt', 'r', encoding='utf-8') as out:
        for line in out:
            line = line.replace("\n", "")
            lines.append(
                line.replace(
                    "</s>",
                    "").replace(
                    "</d>",
                    "")
                )

nlp = spacy.load('./models/spacy-blank-GoogleNews/')

nlp = add_pipes_from_pretrained(nlp)

line_tokens = [[tok for tok in nlp(L)] for L in lines]
line_tokens_with_vector = [[tok for tok in nlp(L) if tok.has_vector] for L in lines]

line_token_lengths = [len(toks) for toks in line_tokens]
line_token_with_vector_lengths = [len(toks) for toks in line_tokens_with_vector]

plt.title('Per-Line Token Counts for Movie Lines')
plt.hist(line_token_lengths, bins=20)
plt.savefig('./dat/preprocess/formatted_movie_lines_spacy_token_counts.png')
plt.close()


plt.title('Per-Line Vectorized Token Counts for Movie Lines')
plt.hist(line_token_with_vector_lengths, bins=20)
plt.savefig('./dat/preprocess/formatted_movie_lines_spacy_token_with_vector_counts.png')
plt.close()


plt.title('Per-Line Token Counts for Movie Lines')
plt.hist(line_token_lengths, bins=20, label='token count')
plt.hist(line_token_with_vector_lengths, bins=20, label='token w/vector count')
plt.legend()
plt.savefig('./dat/preprocess/formatted_movie_lines_spacy_token_counts_compare.png')
plt.close()
