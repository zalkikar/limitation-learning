#!/usr/bin/python

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