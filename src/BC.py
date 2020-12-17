import gensim
import random
import math
import argparse 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import get_model
from models.config import TOKENS_RAW_CUTOFF
from models.seq2seqattn import init_weights, EncRnn, DecRnn, Seq2SeqAttn

from matplotlib import pyplot as plt
from matplotlib import ticker


def display_attention(sentence, translation, attention):
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    
    attention = attention.squeeze(1).cpu().detach().numpy()
    
    cax = ax.matshow(attention, cmap='bone')
   
    ax.tick_params(labelsize=15)
    ax.set_xticklabels(['']+sentence,rotation=45)
    ax.set_yticklabels(['']+translation)
    ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                       rotation=45)
    #ax.set_yticklabels(['']+translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def train(d, model, optimizer, criterion, sos_ind, eos_ind, SEQ_LEN, CLIP, EPOCHS = 10):
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for idx, (index, vects) in enumerate(d.items()):
                # each is N x 300
                input_state, next_state = vects[0], vects[1]
                
                # add <sos> and <eos>
                input_state = torch.cat((torch.LongTensor([sos_ind]), input_state, torch.LongTensor([eos_ind])), dim=0)
                next_state = torch.cat((torch.LongTensor([sos_ind]), next_state, torch.LongTensor([eos_ind])), dim=0)

                trg = next_state.unsqueeze(0)

                optimizer.zero_grad()
                output = model(input_state.unsqueeze(0), torch.Tensor([SEQ_LEN]), trg)


                trg = trg.transpose(1,0)
                ##print("\n")
                ##print(trg.shape) #trg = [trg len, batch size]
                ##print(output.shape) #output = [trg len, batch size, output dim]
                
                output_dim = output.shape[-1]
                
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
                
                ##print(trg.shape) #trg = [(trg len - 1) * batch size]
                ##print(output.shape) #output = [(trg len - 1) * batch size, output dim]
                
                loss = criterion(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()
                epoch_loss += loss.item()

        model.eval()
        ### TODO: RUN ON VALIDATION, record loss
        
        print(f'\tTrain Loss: {epoch_loss / len(d):.3f} | Train PPL: {math.exp(epoch_loss / len(d)):7.3f}')



def translate_sentence(words, input_state, next_state, model, max_len):
    
    model.eval()
    src_tensor = input_state.unsqueeze(0)
    src_len = torch.Tensor([max_len])

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor.transpose(1,0))
    # get first decoder input (<sos>)'s one hot
    trg_indexes = [next_state[0]]
    # create a array to store attetnion
    attentions = torch.zeros(max_len, 1, len(input_state))
    #print(attentions.shape)


    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]])
        #print(trg_tensor.shape)
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
        #print(F.softmax(output))
        attentions[i] = attention
        pred_token = output.argmax(1).item()
        if pred_token == "<eos>": # end of sentence.
            break
        trg_indexes.append(pred_token)
        
    trg_tokens = [words[int(ind)] for ind in trg_indexes]
    #  remove <sos>
    return trg_tokens[1:], attentions[:len(trg_tokens)-1]
    
   

def observe(w2v_model, words, model, d, sos_ind, eos_ind, SEQ_LEN):
    src = None
    trg = None
    for idx, (index, vects) in enumerate(d.items()):
        # each is N x 300
        input_state, next_state = vects[0], vects[1]
        # add <sos> and <eos>
        input_state = torch.cat((torch.LongTensor([sos_ind]), input_state, torch.LongTensor([eos_ind])), dim=0)
        next_state = torch.cat((torch.LongTensor([sos_ind]), next_state, torch.LongTensor([eos_ind])), dim=0)

        src = input_state
        trg = next_state

        print("src = {}".format([words[int(ind)] for ind in src.numpy()]))
        print("trg = {}".format([words[int(ind)] for ind in trg.numpy()]))

        translation, attention = translate_sentence(words, src, trg, model, SEQ_LEN)

        # remove <eos>s (<sos> already removed)
        translation_parsed = [tok for tok in translation if tok != "<eos>"] # too hacky??
        print(f'predicted trg = {translation_parsed}') 

        embedded_action = torch.Tensor([w2v_model.wv[tok] for tok in translation_parsed])
        # drop <sos>, <eos>
        expert_action = torch.Tensor([w2v_model.wv.vectors[ind] for ind in next_state][1:-1]) 

        #print(expert_action.shape)
        #print(embedded_action.shape)

        #display_attention([words[int(ind)] for ind in src.numpy()], translation, attention)

        if idx >= 10:
            break
        print("\n")


   

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', 
                        type=int, 
                        default=50)

    parser.add_argument('--n_hidden', 
                        type=int, 
                        default=64)

    args = parser.parse_args()

    #w2v_model = gensim.models.Word2Vec.load("../models/custom_w2v")
    w2v_model = gensim.models.Word2Vec.load("../models/custom_w2v_intersect_GoogleNews")
    w2v_model.init_sims(replace=True) #precomputed l2 normed vectors in-place – saving the extra RAM
    # random for special marker tokens
    w2v_model.wv["<sos>"] = np.random.rand(300) # MAKE SURE THIS L2 normed
    w2v_model.wv["<eos>"] = np.random.rand(300) # MAKE SURE THIS L2 normed
    # vocab, embed dims
    VOCAB_SIZE, EMBED_DIM = w2v_model.wv.vectors.shape
    # w2ind from w2v
    w2ind = {token: token_index for token_index, token in enumerate(w2v_model.wv.index2word)} 
    # padding token for now
    TRG_PAD_IDX = w2ind["."] # this is 0
    
    sos_ind = w2ind['<sos>']
    eos_ind = w2ind['<eos>']

    SEQ_LEN = 5 + 2 # sos, eos tokens

    d = torch.load('../dat/processed/padded_vectorized_states_v3.pt')

    clip = 1
    device = 'cpu' # for now
    print(SEQ_LEN, VOCAB_SIZE, EMBED_DIM)

    enc = EncRnn(hidden_size=args.n_hidden, num_layers=2, embed_size=EMBED_DIM)
    dec = DecRnn(hidden_size=args.n_hidden, num_layers=2, embed_size=EMBED_DIM, output_size=VOCAB_SIZE)
    model = Seq2SeqAttn(enc, dec, TRG_PAD_IDX, VOCAB_SIZE, device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    model.apply(init_weights)

    train(d, model, optimizer, criterion, sos_ind, eos_ind, SEQ_LEN, clip, args.epochs)

    w2v_model.vocabulary.sorted_vocab
    word_counts = {word: vocab_obj.count for word, vocab_obj in w2v_model.wv.vocab.items()}
    word_counts = sorted(word_counts.items(), key=lambda x:-x[1])
    words = [t[0] for t in word_counts]

    observe(w2v_model, words, model, d, sos_ind, eos_ind, SEQ_LEN)
        

if __name__ == '__main__':
    main()
    
