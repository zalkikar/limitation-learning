import gensim
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#model = gensim.models.Word2Vec.load("../models/custom_w2v")
w2v_model = gensim.models.Word2Vec.load("../models/custom_w2v_intersect_GoogleNews")
w2v_model.init_sims(replace=True) #precomputed l2 normed vectors in-place – saving the extra RAM
# random for special marker tokens
w2v_model.wv["<sos>"] = np.random.rand(300)
w2v_model.wv["<eos>"] = np.random.rand(300)
# vocab, embed dims
VOCAB_SIZE, EMBED_DIM = w2v_model.wv.vectors.shape
# w2ind from w2v
w2ind = {token: token_index for token_index, token in enumerate(w2v_model.wv.index2word)} 
# padding token for now
TRG_PAD_IDX = w2ind["."] # this is 0

d = torch.load('../dat/processed/padded_vectorized_states_v3.pt')




TOKENS_RAW_CUTOFF = 5

def get_vectors():
    model = gensim.models.Word2Vec.load("../models/custom_w2v_intersect_GoogleNews") # ("./models/custom_w2v")
    model.init_sims(replace=True) #precomputed l2 normed vectors in-place – saving the extra RAM
    return torch.FloatTensor(w2v_model.wv.vectors)

def from_pretrained(embeddings=None, freeze=False):
    if not embeddings:
        embeddings = get_vectors() # 2 D embeddings param
    rows, cols = embeddings.shape
    # A simple lookup table that stores embeddings of a fixed dictionary and size.
    embedding = torch.nn.Embedding(num_embeddings=rows, embedding_dim=cols)
    embedding.weight = torch.nn.Parameter(embeddings)
    # no update if freeze=True (default is False)
    embedding.weight.requires_grad = not freeze
    return embedding

SEQ_LEN = 5 + 2 # sos, eos tokens

class EncRnn(nn.Module):
    
    def __init__(self, hidden_size, num_layers,
                 device='cpu', drop_prob=0, lstm=True, feature_norm=False,
                 embed_size=EMBED_DIM,
                 bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.embedding = from_pretrained()

        self.memory_cell = torch.nn.GRU(input_size=embed_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True,
                                # make dropout 0 if num_layers is 1
                                dropout=drop_prob * (num_layers != 1),
                                bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, src_len):
        x = self.dropout(self.embedding(x))
        # packing for computation and performance
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, batch_first=True, lengths = src_len)
        out, hidden = self.memory_cell(packed_x)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True) # unpack
        out = out.transpose(1,0)
        # initial decoder hidden is final hidden state of the forwards and
        # backwards encoder RNNs fed through a linear layer
        concated = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = torch.tanh(self.linear(concated))
        return out, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        
        self.attn = nn.Linear((hidden_size * 2) + hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10) # using masking, we can force the attention to only be over non-padding elements.
        return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers,
                 device='cpu', drop_prob=0, lstm=True, feature_norm=False,
                 embed_size=EMBED_DIM,output_size=VOCAB_SIZE,
                 bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.attention = Attention(hidden_size)

        self.embedding = from_pretrained()
        """
        self.memory_cell = torch.nn.GRU(input_size=(hidden_size*2)+embed_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                # make dropout 0 if num_layers is 1
                                dropout=drop_prob * (num_layers != 1),
                                bidirectional=False)
        """
        self.memory_cell = torch.nn.GRU((hidden_size * 2) + embed_size, hidden_size)
        self.linear = nn.Linear((hidden_size * 3)+embed_size, output_size)
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, input, hidden, encoder_outputs, mask):
             
        ##print(input.shape) #input = [batch size]
        ##print(hidden.shape) #hidden = [batch size, dec hid dim]
        ##print(encoder_outputs.shape) #encoder_outputs = [src len, batch size, enc hid dim * 2]
        ##print(mask.shape) #mask = [batch size, src len]
        
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        ##print(embedded.shape)
        a = self.attention(hidden, encoder_outputs, mask)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        ##print(encoder_outputs.shape)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        ##print(weighted.shape)
        dec_input = torch.cat((embedded, weighted), dim = 2)
        ##print(dec_input.shape, hidden.unsqueeze(0).shape)
        output, hidden = self.memory_cell(dec_input, hidden.unsqueeze(0))

        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.linear(torch.cat((output, weighted, embedded), dim = 1))
        ##print(prediction.shape)
        return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        
    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0) ### used to be != ???
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        src = src.transpose(1,0)
        trg = trg.transpose(1,0)

        ##print(src.shape) #src = [src len, batch size]
        ##print(src_len.shape) #src_len = [batch size]
        ##print(trg.shape) #trg = [trg len, batch size]
        
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
                    
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = VOCAB_SIZE 
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        ##print(outputs)
        
        src = src.transpose(1,0)
        encoder_outputs, hidden = self.encoder(src, src_len)
        src = src.transpose(1,0)
        ##print(encoder_outputs, hidden)

        input = trg[0,:]
        ##print(input)
        
        mask = self.create_mask(src)
        #print(f'src = {src}')
        #print(f'mask = {mask}')

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            #print(input, top1)
            input = trg[t] if teacher_force else top1
            
        return outputs

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
def translate_sentence(words, input_state, next_state, model, max_len = SEQ_LEN):
    
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
        #print(F.softmax(output))
        attentions[i] = attention
        pred_token = output.argmax(1).item()
        if pred_token == "<eos>": # end of sentence.
            break
        trg_indexes.append(pred_token)
        
    trg_tokens = [words[int(ind)] for ind in trg_indexes]
    #  remove <sos>
    return trg_tokens[1:], attentions[:len(trg_tokens)-1]


N_HIDDEN = 64
clip = 1
device = 'cpu' # for now
print(SEQ_LEN, VOCAB_SIZE, EMBED_DIM)

enc = EncRnn(hidden_size=N_HIDDEN, num_layers=2, embed_size=EMBED_DIM)
dec = Decoder(hidden_size=N_HIDDEN, num_layers=2, embed_size=EMBED_DIM, output_size=VOCAB_SIZE)
model = Seq2Seq(enc, dec, TRG_PAD_IDX, device)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

model.apply(init_weights)

sos_ind = w2ind['<sos>']
eos_ind = w2ind['<eos>']

model.train()
EPOCHS = 5
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
    print(f'\tTrain Loss: {epoch_loss / len(d):.3f} | Train PPL: {math.exp(epoch_loss / len(d)):7.3f}')


w2v_model.vocabulary.sorted_vocab
word_counts = {word: vocab_obj.count for word, vocab_obj in w2v_model.wv.vocab.items()}
word_counts = sorted(word_counts.items(), key=lambda x:-x[1])
words = [t[0] for t in word_counts]


src = None
trg = None
for idx, (index, vects) in enumerate(d.items()):
        # each is N x 300
        input_state, next_state = vects[0], vects[1]
        # raw strings corresponding to embeddings
        raw_input_state, raw_next_state = list(raw.keys())[index], raw[list(raw.keys())[index]]
        #print(input_state, next_state)
        
        # add <sos> and <eos>
        input_state = torch.cat((torch.LongTensor([sos_ind]), input_state, torch.LongTensor([eos_ind])), dim=0)
        next_state = torch.cat((torch.LongTensor([sos_ind]), next_state, torch.LongTensor([eos_ind])), dim=0)

        src = input_state
        trg = next_state

        print("src = {}".format([words[int(ind)] for ind in src.numpy()]))
        print("trg = {}".format([words[int(ind)] for ind in trg.numpy()]))

        translation, attention = translate_sentence(words, src,trg, model)

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
