import srsly

"""
Pre-train the “token to vector” (tok2vec) layer of pipeline components, using an approximate language-modeling objective. 
Specifically, we load pretrained vectors, and train a component like a CNN, BiLSTM, etc to predict vectors which match the pretrained ones. 
The weights are saved to a directory after each epoch. 
You can then pass a path to one of these pretrained weights files to the spacy train command. 
You can try to use a few with low Loss values reported in the output.
loss function is either L2 or cosine distance
"""

tok_data = []
with open('./dat/processed/formatted_movie_lines.txt', 'r', encoding = 'utf-8') as pf:
    for line in pf:
        line = line.replace("\n", "").replace("</s>","").replace("</d>","")
        toks = line.strip().split(' ') # already processed so split by space
        tok_data.append({'tokens':toks})

srsly.write_jsonl("./dat/processed/formatted_movie_lines.jsonl", tok_data)
