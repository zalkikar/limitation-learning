# -*- coding: utf-8 -*-

import csv
import spacy
import time
from config import SPACY_MODEL_TYPE
import pandas as pd
from ast import literal_eval
import re
import unicodedata


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def loadTxt(fileType, fields):

    d = pd.read_csv(f'./dat/cornell-movie-dialog-corpus/movie_{fileType}.txt',
                    engine = 'python',
                    sep=" \+\+\+\$\+\+\+ ",
                    encoding='iso-8859-1',
                    header=None) # faster than open()

    d.columns = fields

    if fileType == 'conversations': # type list for utterance IDs
        d[fields[-1]] = d[fields[-1]].apply(literal_eval)

    return d


def preprocessTxt(nlp, lines, conversations):
    """
    these: </u> and <u> were present in raw text
    as a quick fix and to avoid a rerun, we handle this later
    by dropping "< u > " and "</u >"
    also "\x92" was present, iso8859 encoding to utf-8 issues....
    makeshift solution for now
    """
    convoLines = []
    #rawLines = []

    for ind,uts in enumerate(conversations.utteranceIDs.values):

        texts = [lines.loc[ lines.lineID == ut , 'text' ].values[0] for ut in uts]
        #texts = [normalizeString(text.decode('iso-8859-1').encode('utf8')) for text in texts]
        textTokens = [' '.join([tok.text if not tok.ent_type_ else f"<{tok.ent_type_.lower()}>" for tok in nlp(text.strip())]) for text in texts if text]

        assert len([t for t in texts if t]) == len(textTokens)

        convoLines.append( "\n".join ( [" ".join(textTokens[i].split())+" </d>" if i+1==len(textTokens) else " ".join(textTokens[i].split())+" </s>" for i in range(len(textTokens))] ) )

        #texts = [text for text in texts if text]
        #rawLines.append("\n".join ( [" ".join(texts[i].split())+" </d>" if i+1==len(texts) else " ".join(texts[i].split())+" </s>" for i in range(len(textTokens))]))
    return convoLines#, rawLines



def run_preprocess():

    print('loading lines...')
    start = time.process_time()
    lines = loadTxt('lines',
                    ["lineID", "characterID", "movieID", "character", "text"])
    print(f'complete {time.process_time() - start}')

    print('loading conversations...')
    start = time.process_time()
    conversations = loadTxt('conversations',
                            ["character1ID", "character2ID", "movieID", "utteranceIDs"])
    print(f'complete {time.process_time() - start}')

    nlp = spacy.load(SPACY_MODEL_TYPE) # spacy ner

    """tokenizer = nlp.Defaults.create_tokenizer(nlp)""" # tokenizer with default punct rules & exceptions for transfer NER vocabulary
    #tokenizer = nlp.tokenizer # for now we just go with the learned transfer tokenization
    
    print("preprocessing...")
    start = time.process_time()

    #preprocessLines, rawLines = preprocessTxt(nlp, lines, conversations)
    preprocessLines = preprocessTxt(nlp, lines, conversations)

    print(f'complete {time.process_time() - start}')
    
    with open('./dat/processed/formatted_movie_lines.txt', 'w', encoding='utf-8') as outFile:
        outFile.write("\n".join( preprocessLines ))
        
    #with open('./dat/processed/raw_movie_lines.txt', 'w', encoding='iso-8859-1') as outFile:
    #    outFile.write("\n".join( rawLines ))

    
if __name__ == "__main__":
    run_preprocess()
    #quick_run_check()