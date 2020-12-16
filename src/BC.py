import gensim
import random
import math
import argparse 

import torch
import torch.nn as nn

from models.utils import get_model
from models.config import TOKENS_RAW_CUTOFF
from models.seq2seqattn import init_weights, EncRnn, DecRnn, Seq2SeqAttn

from matplotlib import pyplot as plt
from matplotlib import ticker


def train(d, model, optimizer, criterion, SEQ_LEN, CLIP, EPOCHS = 10):
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for idx, (index, vects) in enumerate(d.items()):
                # each is N x 300
                input_state, next_state = vects[0], vects[1]
                ### add <sos> and <eos>
                ##tokens = [src_field.init_token] + tokens + [src_field.eos_token]

                trg = next_state.unsqueeze(0)

                optimizer.zero_grad()
                output = model(input_state.unsqueeze(0), torch.Tensor([SEQ_LEN]), trg)


                trg = trg.transpose(1,0)
                ##print("\n")
                ##print(trg.shape) #trg = [trg len, batch size]
                ##print(output.shape) #output = [trg len, batch size, output dim]
                
                output_dim = output.shape[-1]
                
                output = output.view(-1, output_dim) #output[1:].view(-1, output_dim)
                trg = trg.view(-1) #trg[1:].view(-1)
                
                ##print(trg.shape) #trg = [(trg len - 1) * batch size]
                ##print(output.shape) #output = [(trg len - 1) * batch size, output dim]
                
                loss = criterion(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()
                epoch_loss += loss.item()
        print(f'\tTrain Loss: {epoch_loss / len(d):.3f} | Train PPL: {math.exp(epoch_loss / len(d)):7.3f}')


def translate_sentence(words, input_state, next_state, model, SEQ_LEN, max_len = TOKENS_RAW_CUTOFF):
    
    model.eval()
    src_tensor = input_state.unsqueeze(0)
    src_len = torch.Tensor([SEQ_LEN])

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
        attentions[i] = attention
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        #if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
        #    break
        
    trg_tokens = [words[int(ind)] for ind in trg_indexes]
    #return trg_tokens[1:], attentions[:len(trg_tokens)-1]
    return trg_tokens, attentions[:len(trg_tokens)]


def display_attention(sentence, translation, attention):
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    
    attention = attention.squeeze(1).cpu().detach().numpy()
    
    cax = ax.matshow(attention, cmap='bone')
   
    ax.tick_params(labelsize=15)
    ax.set_xticklabels(['']+sentence,rotation=45)
    ax.set_yticklabels(['']+translation)
    #ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
    #                   rotation=45)
    #ax.set_yticklabels(['']+translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def observe(words, model, src, trg, SEQ_LEN):

    print("src = {}".format(" ".join([words[int(ind)] for ind in src.numpy()])))
    print("trg = {}".format(" ".join([words[int(ind)] for ind in trg.numpy()])))

    translation, attention = translate_sentence(words, src, trg, model, SEQ_LEN)

    print(f'predicted trg = {" ".join(translation)}')
    #display_attention([words[int(ind)] for ind in src.numpy()], translation, attention)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', 
                        type=int, 
                        default=20)

    parser.add_argument('--n_hidden', 
                        type=int, 
                        default=64)

    args = parser.parse_args()

    d = torch.load('../dat/processed/padded_vectorized_states_v3.pt')

    w2v_model = get_model() # already normed

    N_HIDDEN = args.n_hidden
    SEQ_LEN = TOKENS_RAW_CUTOFF
    VOCAB_SIZE, EMBED_DIM = w2v_model.wv.vectors.shape

    CLIP = 1 # for exploding gradients
    device = 'cpu'
    enc = EncRnn(hidden_size=N_HIDDEN, num_layers=2, embed_size=EMBED_DIM)
    dec = DecRnn(hidden_size=N_HIDDEN, num_layers=2, embed_size=EMBED_DIM, output_size=VOCAB_SIZE)

    TRG_PAD_IDX = 0
    model = Seq2SeqAttn(enc, dec, TRG_PAD_IDX, device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX) # 0 ind padding ignore
                
    model.apply(init_weights)

    train(d, model, optimizer, criterion, SEQ_LEN, CLIP, args.epochs)

    torch.save(model.state_dict(), './generators/seq2seq-model.pt')

    assert w2v_model.vocabulary.sorted_vocab == True
    word_counts = {word: vocab_obj.count for word, vocab_obj in w2v_model.wv.vocab.items()}
    word_counts = sorted(word_counts.items(), key=lambda x:-x[1])
    words = [t[0] for t in word_counts]

    keys = random.sample(list(d.keys()), 20)
    print("\n")
    for k in keys:
        input_state, next_state = d[k]
        observe(words, model, input_state, next_state, SEQ_LEN)
        print("\n")

if __name__ == '__main__':
    main()
    
