from preprocess import run_preprocess
from embeds_spacy import create_google_news_vectors
from dialog_states import run_dialog_states
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--vectors_path",
                        type=str,
                        required=True)

    args = parser.parse_args()
    
    """ completed in prior slurm job """
    #print("running preprocess...")
    #run_preprocess()

    print('loading GoogleNewsVectors into blank spacy...')
    create_google_news_vectors(args.vectors_path)

    print('creating dialogue states (raw, vectorized, padded)...')
    run_dialog_states()
