import spacy
from spacy.attrs import ORTH
from config import SPACY_MODEL_TYPE

def add_pipes_from_pretrained(nlp):

    nlp_pretrained = spacy.load(SPACY_MODEL_TYPE)  # spacy ner
    model_tags = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 
                  'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

    # google news vector based tokenization
    #nlp.tokenizer = Tokenizer(nlp.vocab)

    # novel tokenization algorithm specific to pretrained spacy model, non-destructive, not a model
    nlp.tokenizer = nlp_pretrained.tokenizer

    # tokenization exceptions
    special_cases = [[{ORTH: f"<{t.lower()}>"}] for t in model_tags]
    special_tags = [f"<{t.lower()}>" for t in model_tags]
    for t,c in zip(special_tags,special_cases):
        nlp.tokenizer.add_special_case(t, c)

    ### print([tok.text for tok in nlp("This a test to see if <person> isn't split up.")])

    # learned dependency parsing model, implements thinc.neural.Model API
    # can lead to problems w/out retraining, but we might not care about sentence spans so we don't need it
    #nlp.add_pipe(nlp_pretrained.get_pipe('parser'), name='parser')
    
    return nlp