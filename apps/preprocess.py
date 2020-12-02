# -*- coding: utf-8 -*-

from utils import loadTxt
import csv
import spacy
import time

def preprocessTxt(nlp, lines, conversations):
    convoLines = []

    for ind,uts in enumerate(conversations.utteranceIDs.values):

        texts = [lines.loc[ lines.lineID == ut , 'text' ].values[0] for ut in uts]
        textTokens = [' '.join([tok.text if not tok.ent_type_ else f"<{tok.ent_type_.lower()}>" for tok in nlp(text.strip())]) for text in texts if text]

        assert len([t for t in texts if t]) == len(textTokens)

        convoLines.append( "\n".join ( [" ".join(textTokens[i].split())+" </d>" if i+1==len(textTokens) else " ".join(textTokens[i].split())+" </s>" for i in range(len(textTokens))] ) )

    return convoLines


def main():

    start = time.process_time()
    lines = loadTxt('lines',
                    ["lineID", "characterID", "movieID", "character", "text"])
    print(time.process_time() - start)
    start = time.process_time()
    conversations = loadTxt('conversations',
                            ["character1ID", "character2ID", "movieID", "utteranceIDs"])
    print(time.process_time() - start)

    nlp = spacy.load('en_core_web_lg') # spacy ner

    """tokenizer = nlp.Defaults.create_tokenizer(nlp)""" # tokenizer with default punct rules & exceptions for transfer NER vocabulary
    #tokenizer = nlp.tokenizer # for now we just go with the learned transfer tokenization
    
    start = time.process_time()
    preprocessLines = preprocessTxt(nlp, lines, conversations[0:500])
    print(time.process_time() - start)
    
    with open('./dat/preprocess/formatted_movie_lines.txt', 'w', encoding='utf-8') as outFile:
        outFile.write("\n".join( preprocessLines ))

    
if __name__ == "__main__":
    main()