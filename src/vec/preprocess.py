# -*- coding: utf-8 -*-

import csv
import spacy
import time
from config import SPACY_MODEL_TYPE
import pandas as pd
from ast import literal_eval


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
    convoLines = []

    for ind,uts in enumerate(conversations.utteranceIDs.values):

        texts = [lines.loc[ lines.lineID == ut , 'text' ].values[0] for ut in uts]
        textTokens = [' '.join([tok.text if not tok.ent_type_ else f"<{tok.ent_type_.lower()}>" for tok in nlp(text.strip())]) for text in texts if text]

        assert len([t for t in texts if t]) == len(textTokens)

        convoLines.append( "\n".join ( [" ".join(textTokens[i].split())+" </d>" if i+1==len(textTokens) else " ".join(textTokens[i].split())+" </s>" for i in range(len(textTokens))] ) )

    return convoLines


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
    preprocessLines = preprocessTxt(nlp, lines, conversations)
    print(f'complete {time.process_time() - start}')
    
    with open('./dat/processed/formatted_movie_lines.txt', 'w', encoding='utf-8') as outFile:
        outFile.write("\n".join( preprocessLines ))

    
if __name__ == "__main__":
    run_preprocess()
