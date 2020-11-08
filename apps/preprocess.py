#!/usr/bin/python

from utils import loadTxt
import csv
import spacy

def preprocessTxt(tokenizer, lines, conversations):
    convoLines = []

    for ind,uts in enumerate(conversations.utteranceIDs.values):

        texts = [lines.loc[ lines.lineID == ut , 'text' ].values[0] for ut in uts]
        textTokens = [' '.join([tok.text for tok in tokenizer(text.strip())]) for text in texts if text]

        assert len([t for t in texts if t]) == len(textTokens)

        convoLines.append( "\n".join ( [textTokens[i]+" </d>" if i+1==len(textTokens) else textTokens[i]+" </s>" for i in range(len(textTokens))] ) )

    return convoLines


def main():

    lines = loadTxt('lines',
                    ["lineID", "characterID", "movieID", "character", "text"])

    conversations = loadTxt('conversations',
                            ["character1ID", "character2ID", "movieID", "utteranceIDs"])

    nlp = spacy.load('en_core_web_lg') # spacy ner

    """tokenizer = nlp.Defaults.create_tokenizer(nlp)""" # tokenizer with default punct rules & exceptions for transfer NER vocabulary
    tokenizer = nlp.tokenizer # for now we just go with the transfer tokenization

    preprocessLines = preprocessTxt(tokenizer, lines, conversations)

    with open('./dat/preprocess/formatted_movie_lines.txt', 'w', encoding='utf-8') as outFile:
        outFile.write("\n".join( preprocessLines ))

    
if __name__ == "__main__":
    main()