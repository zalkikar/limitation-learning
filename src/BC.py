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

from GAIL import get_cosine_sim

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


def train(train_d, valid_d, w2v_model, words, model, optimizer, criterion, sos_ind, eos_ind, TRG_PAD_IDX, SEQ_LEN, CLIP, device, EPOCHS = 10):

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for idx, (index, vects) in enumerate(train_d.items()):
                # each is N x 300
                input_state, next_state = vects[0], vects[1]
                
                # add <sos> and <eos>
                input_state = torch.cat((torch.LongTensor([sos_ind]), input_state, torch.LongTensor([eos_ind])), dim=0).to(device)
                next_state = torch.cat((torch.LongTensor([sos_ind]), next_state, torch.LongTensor([eos_ind])), dim=0).to(device)

                trg = next_state.unsqueeze(0).to(device)
                seq_len_tensor = torch.Tensor([SEQ_LEN]).to(device)

                optimizer.zero_grad()
                output = model(input_state.unsqueeze(0), seq_len_tensor, trg)


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
                epoch_loss += loss.detach().cpu().item()

        # Train PPL: {math.exp(epoch_loss / len(train_d)):7.3f}
        print(f'\t\tEpoch {epoch+1} Train Loss: {epoch_loss / len(train_d):.3f}')
        evaluate(valid_d, w2v_model, words, model, criterion, sos_ind, eos_ind, TRG_PAD_IDX, SEQ_LEN, device)
        torch.save(model.state_dict(), f'./generators/model-epoch{epoch+1}.pt')


def evaluate(d, w2v_model, words, model, criterion, sos_ind, eos_ind, TRG_PAD_IDX, SEQ_LEN, device, type='Valid'):
    model.eval()
    val_loss = 0
    cos_sims = 0
    with torch.no_grad():
        for idx, (index, vects) in enumerate(d.items()):

            input_state, next_state = vects[0], vects[1]
            input_state = torch.cat((torch.LongTensor([sos_ind]), input_state, torch.LongTensor([eos_ind])), dim=0).to(device)
            next_state = torch.cat((torch.LongTensor([sos_ind]), next_state, torch.LongTensor([eos_ind])), dim=0).to(device)
            trg = next_state.unsqueeze(0).to(device)
            seq_len_tensor = torch.Tensor([int(SEQ_LEN)])

            output = model(input_state.unsqueeze(0), seq_len_tensor, trg)

            trg = trg.transpose(1,0)
            output_dim = output.shape[-1]                
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            val_loss += loss.detach().cpu().item()

            translation, attention = translate_sentence(words, input_state, next_state, model, eos_ind, SEQ_LEN, device)

            # drop <sos>, <eos>
            expert_act = [words[int(ind)] for ind in next_state.numpy()][1:-1]
            # drop multiple instances of padded token
            expert_act_unpadded = []
            for tok in expert_act:
                expert_act_unpadded.append(tok)
                if tok == words[int(TRG_PAD_IDX)]:
                    break
            vectorized_expert_act = [w2v_model.wv[tok] for tok in expert_act_unpadded]
            vectorized_pred_act = [w2v_model.wv[tok] for tok in translation]
            cos_sims += get_cosine_sim(vectorized_expert_act.detach(), vectorized_pred_act.detach(), type = None, seq_len = 5, dim = 300)
    print(f'\t{type} Avg Loss: {val_loss / len(d):.3f} | {type} Avg Cosine Sim: {cos_sims / len(d):.3f}')


def translate_sentence(words, input_state, next_state, model, eos_ind, max_len, device):
    
    model.eval()
    src_tensor = input_state.unsqueeze(0).to(device)
    src_len = torch.Tensor([int(max_len)])

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor.transpose(1,0)).to(device)
    # get first decoder input (<sos>)'s one hot
    trg_indexes = [next_state[0]]
    # create a array to store attetnion
    attentions = torch.zeros(max_len, 1, len(input_state))
    #print(attentions.shape)


    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        #print(trg_tensor.shape)
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
        #print(F.softmax(output))
        attentions[i] = attention
        pred_token = output.argmax(1).item()
        if pred_token == eos_ind: # end of sentence.
            break
        trg_indexes.append(pred_token)
        
    trg_tokens = [words[int(ind)] for ind in trg_indexes]
    #  remove <sos>
    return trg_tokens[1:], attentions[:len(trg_tokens)-1]
    
   

def observe(w2v_model, words, model, d, sos_ind, eos_ind, TRG_PAD_IDX, SEQ_LEN):
    src = None
    trg = None
    for idx, (index, vects) in enumerate(d.items()):
        print("\n")
        # each is N x 300
        input_state, next_state = vects[0], vects[1]
        # add <sos> and <eos>
        input_state = torch.cat((torch.LongTensor([sos_ind]), input_state, torch.LongTensor([eos_ind])), dim=0)
        next_state = torch.cat((torch.LongTensor([sos_ind]), next_state, torch.LongTensor([eos_ind])), dim=0)

        src = input_state
        trg = next_state

        translation, attention = translate_sentence(words, src, trg, model, eos_ind, SEQ_LEN)

        # drop <sos>, <eos>
        expert_act = [words[int(ind)] for ind in next_state.numpy()][1:-1]
        init_act = [words[int(ind)] for ind in input_state.numpy()][1:-1]
        # drop multiple instances of padded token
        expert_act_unpadded, init_act_unpadded = [],[]
        for tok in expert_act:
            expert_act_unpadded.append(tok)
            if tok == words[int(TRG_PAD_IDX)]:
                break
        for tok in init_act:
            init_act_unpadded.append(tok)
            if tok == words[int(TRG_PAD_IDX)]:
                break
        vectorized_expert_act = [w2v_model.wv[tok] for tok in expert_act_unpadded]
        vectorized_pred_act = [w2v_model.wv[tok] for tok in translation]
        cos_sim = get_cosine_sim(vectorized_expert_act, vectorized_pred_act, type = None, seq_len = 5, dim = 300)

        print(f'initial action = {" ".join(init_act_unpadded)}')
        print(f'expert action = {" ".join(expert_act_unpadded)}')
        print(f'predicted action = {" ".join(translation)}')
        print(f'cosine sim {cos_sim}') 

        #display_attention([words[int(ind)] for ind in src.numpy()], translation, attention)

        if idx >= 10:
            break


   

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', 
                        type=int, 
                        default=50)

    parser.add_argument('--n_hidden', 
                        type=int, 
                        default=64)

    parser.add_argument('--n_layers',
                        type=int,
                        default=2)

    args = parser.parse_args()

    w2v_model = get_model()
    # vocab, embed dims
    VOCAB_SIZE, EMBED_DIM = w2v_model.wv.vectors.shape
    # w2ind from w2v
    w2ind = {token: token_index for token_index, token in enumerate(w2v_model.wv.index2word)} 
    # padding token for now
    TRG_PAD_IDX = w2ind["."] # this is 0
    # sentence marker token inds
    sos_ind = w2ind['<sos>']
    eos_ind = w2ind['<eos>']
    # adjusted sequence length
    SEQ_LEN = 5 + 2 # sos, eos tokens
    # padded vectorized states of token indexes
    d = torch.load('../dat/processed/padded_vectorized_states_v3.pt')
    # train test valid split
    train_d = {}
    test_d = {}
    valid_d = {}
    for index, vects in d.items():
        if torch.rand(1) < 0.2:
            test_d[index] = vects
        elif torch.rand(1) < 0.4:
            valid_d[index] = vects
        else:
            train_d[index] = vects
    print(f'train % = {len(train_d)/len(d)}')
    print(f'test % = {len(test_d)/len(d)}')
    print(f'valid % = {len(valid_d)/len(d)}\n')

    clip = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc = EncRnn(hidden_size=args.n_hidden, num_layers=args.n_layers, embed_size=EMBED_DIM)
    dec = DecRnn(hidden_size=args.n_hidden, num_layers=args.n_layers, embed_size=EMBED_DIM, output_size=VOCAB_SIZE)
    model = Seq2SeqAttn(enc, dec, TRG_PAD_IDX, VOCAB_SIZE, device).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX).to(device)

    assert w2v_model.vocabulary.sorted_vocab == True
    word_counts = {word: vocab_obj.count for word, vocab_obj in w2v_model.wv.vocab.items()}
    word_counts = sorted(word_counts.items(), key=lambda x:-x[1])
    words = [t[0] for t in word_counts]

    model.apply(init_weights)
    train(train_d, valid_d, w2v_model, words, model, optimizer, criterion, sos_ind, eos_ind, TRG_PAD_IDX, SEQ_LEN, clip, device, args.epochs)
    
    evaluate(test_d, w2v_model, words, model, criterion, sos_ind, eos_ind, TRG_PAD_IDX, SEQ_LEN, device, type='Test')
    
    observe(w2v_model, words, model, d, sos_ind, eos_ind, TRG_PAD_IDX, SEQ_LEN)
        

if __name__ == '__main__':
    main()
    
