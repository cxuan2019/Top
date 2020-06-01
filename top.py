#!/usr/bin/env python
# coding: utf-8

# # Stanford XCS224U Final Project - Top
# 

# In[171]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator, TabularDataset

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np
import pandas as pd

import random
import math
import time
import sys
import gc

model = None
learn = None
gc.collect()

# Next, we'll set the random seed for reproducability.

# In[177]:


MODE = sys.argv[1]  # 'sketch'  # 'lotv', 'compact', 'mix'

print("RUN Mode: {}".format(MODE))
# exit()

train_file = 'train_' + MODE + '.tsv'
valid_file = 'eval_' + MODE + '.tsv'
test_file = 'test_' + MODE + '.tsv'

model_file = 'top_' + MODE + '.pt'

SEED = 2345 #1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# As before, we'll import spaCy and define the German and English tokenizers.

# In[173]:


# spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


# In[174]:


def tokenize_trg(text):
    """
    Tokenizes parse
    """
    return text.split()

def tokenize_src(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


# When using packed padded sequences, we need to tell PyTorch how long the actual (non-padded) sequences are. Luckily for us, TorchText's `Field` objects allow us to use the `include_lengths` argument, this will cause our `batch.src` to be a tuple. The first element of the tuple is the same as before, a batch of numericalized source sentence as a tensor, and the second element is the non-padded lengths of each source sentence within the batch.

# In[175]:


SRC = Field(tokenize = tokenize_src, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            include_lengths = True)

TRG = Field(tokenize = tokenize_trg, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = False)


# We then load the data.

# In[178]:


# train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
#                                                     fields = (SRC, TRG))
top_datafields = [('src', SRC), ('trg', TRG)]
train_data, valid_data, test_data = TabularDataset.splits(path='data', train=train_file, validation=valid_file,
                                                          test=test_file, format='tsv', skip_header=False, 
                                                          fields=top_datafields)
                                                        


# And build the vocabulary.

# In[179]:


SRC.build_vocab(train_data, vectors="glove.6B.200d")
TRG.build_vocab(train_data, min_freq = 1)


# Next, we handle the iterators.
# 
# One quirk about packed padded sequences is that all elements in the batch need to be sorted by their non-padded lengths in descending order, i.e. the first sentence in the batch needs to be the longest. We use two arguments of the iterator to handle this, `sort_within_batch` which tells the iterator that the contents of the batch need to be sorted, and `sort_key` a function which tells the iterator how to sort the elements in the batch. Here, we sort by the length of the `src` sentence.

# In[180]:


BATCH_SIZE = 128

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
     batch_size = BATCH_SIZE,
     sort_within_batch = True,
     sort_key = lambda x : len(x.src),
     device = device)


# ## Building the Model
# 
# ### Encoder
# 
# Next up, we define the encoder.
# 
# The changes here all within the `forward` method. It now accepts the lengths of the source sentences as well as the sentences themselves. 
# 
# After the source sentence (padded automatically within the iterator) has been embedded, we can then use `pack_padded_sequence` on it with the lengths of the sentences. `packed_embedded` will then be our packed padded sequence. This can be then fed to our RNN as normal which will return `packed_outputs`, a packed tensor containing all of the hidden states from the sequence, and `hidden` which is simply the final hidden state from our sequence. `hidden` is a standard tensor and not packed in any way, the only difference is that as the input was a packed sequence, this tensor is from the final **non-padded element** in the sequence.
# 
# We then unpack our `packed_outputs` using `pad_packed_sequence` which returns the `outputs` and the lengths of each, which we don't need. 
# 
# The first dimension of `outputs` is the padded sequence lengths however due to using a packed padded sequence the values of tensors when a padding token was the input will be all zeros.

# In[181]:


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        
        #src = [src len, batch size]
        #src_len = [batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
                
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)
                
        packed_outputs, hidden = self.rnn(packed_embedded)
                                 
        #packed_outputs is a packed sequence containing all hidden states
        #hidden is now from the final non-padded element in the batch
            
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 
            
        #outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros
            
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        return outputs, hidden


    def load_pretrained_embeddings(self, vocab):
        self.embedding.weight.data.copy_(vocab.vectors)


# ### Attention
# 
# The attention module is where we calculate the attention values over the source sentence. 
# 
# Previously, we allowed this module to "pay attention" to padding tokens within the source sentence. However, using *masking*, we can force the attention to only be over non-padding elements.
# 
# The `forward` method now takes a `mask` input. This is a **[batch size, source sentence length]** tensor that is 1 when the source sentence token is not a padding token, and 0 when it is a padding token. For example, if the source sentence is: ["hello", "how", "are", "you", "?", `<pad>`, `<pad>`], then the mask would be [1, 1, 1, 1, 1, 0, 0].
# 
# We apply the mask after the attention has been calculated, but before it has been normalized by the `softmax` function. It is applied using `masked_fill`. This fills the tensor at each element where the first argument (`mask == 0`) is true, with the value given by the second argument (`-1e10`). In other words, it will take the un-normalized attention values, and change the attention values over padded elements to be `-1e10`. As these numbers will be miniscule compared to the other values they will become zero when passed through the `softmax` layer, ensuring no attention is payed to padding tokens in the source sentence.

# In[182]:


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
  
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention = [batch size, src len]
        
        attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim = 1)


# ### Decoder
# 
# The decoder only needs a few small changes. It needs to accept a mask over the source sentence and pass this to the attention module. As we want to view the values of attention during inference, we also return the attention tensor.

# In[183]:


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
             
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        #mask = [batch size, src len]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs, mask)
                
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0), a.squeeze(1)


# ### Seq2Seq
# 
# The overarching seq2seq model also needs a few changes for packed padded sequences, masking and inference. 
# 
# We need to tell it what the indexes are for the pad token and also pass the source sentence lengths as input to the `forward` method.
# 
# We use the pad token index to create the masks, by creating a mask tensor that is 1 wherever the source sentence is not equal to the pad token. This is all done within the `create_mask` function.
# 
# The sequence lengths as needed to pass to the encoder to use packed padded sequences.
# 
# The attention at each time-step is stored in the `attentions` 

# In[184]:


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        
    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #src_len = [batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
                    
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)
                
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        mask = self.create_mask(src)

        #mask = [batch size, src len]
                
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state, all encoder hidden states 
            #  and mask
            #receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
            
        return outputs


# ## Training the Seq2Seq Model
# 
# Next up, initializing the model and placing it on the GPU.

# In[185]:


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 200 # 256
DEC_EMB_DIM = 128
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5 #0.5
DEC_DROPOUT = 0.5
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
LEARNING_RATE = 0.001

print(INPUT_DIM, OUTPUT_DIM)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
enc.load_pretrained_embeddings(SRC.vocab)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)


# Then, we initialize the model parameters.

# In[186]:


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
model.apply(init_weights)


# We'll print out the number of trainable parameters in the model, noticing that it has the exact same amount of parameters as the model without these improvements.

# In[187]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# Then we define our optimizer and criterion. 
# 
# The `ignore_index` for the criterion needs to be the index of the pad token for the target language, not the source language.

# In[188]:


optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# In[189]:


TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


# In[190]:


# print(vars(train_data.examples[0]))


# In[191]:


def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    epoch_accuracy = 0
    
    for i, batch in enumerate(iterator):
        
        src, src_len = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, src_len, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        epoch_accuracy += compute_accuracy(output, trg)
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_accuracy / len(iterator), epoch_loss / len(iterator)


# In[192]:



def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    epoch_accuracy = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):

            src, src_len = batch.src
            trg = batch.trg

            output = model(src, src_len, trg, 0) #turn off teacher forcing
            
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            epoch_accuracy += compute_accuracy(output, trg)
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_accuracy / len(iterator), epoch_loss / len(iterator)


# Then, we'll define a useful function for timing how long epochs take.

# In[193]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# The penultimate step is to train our model. Notice how it takes almost half the time as our model without the improvements added in this notebook.

# In[194]:


TRG_EOS_IDX = SRC.vocab.stoi['<eos>']

def convert_seq(seq, vocab): # conver sequence from ids to tokens.
    seq = seq.permute(1, 0) # [batch, seq len]
    seq = seq.tolist()
    
    def get_text(l, vocab):
        ret = [vocab.itos[item] for item in l]
        return ' '.join(ret)
    
    seq = [get_text(seq[i], vocab) for i in range(0, len(seq))]
    return seq
        
    
def compute_accuracy(output, trg):
    #trg = [(trg len - 1) , batch size]
    #output = [(trg len - 1) , batch size, output dim]
    global SRC
    global TRG

    top1 = output.argmax(-1) #  [(trg len - 1), batch size]
    batch_size = trg.size()[1]
    
    merged = list(zip(convert_seq(trg, TRG.vocab), convert_seq(top1, TRG.vocab)))
    
    pre = top1.permute(1, 0).tolist() # [batch, trg len - 1]
    gold = trg.permute(1, 0).tolist()
    
    def compare_lists(l1, l2): #l2 is gold
        if len(l1) != len(l2):
            return 0
        
        inds = range(0, l2.index(TRG_EOS_IDX))
        
        ret = [abs(l1[i] - l2[i]) for i in inds[1:]]
        return 1 if sum(ret) == 0 else 0
    
    result = sum([compare_lists(pre[i], gold[i]) for i in range(0, batch_size)])
       
    return 100 * result / batch_size


# In[108]:


N_EPOCHS = 100
top_k = 5
report_iter = 10

CLIP = 1

best_valid_accuracy = 0
# best_valid_loss = float('inf')
# best_epoch = -1

sorted_results = []
available_model_id = 0
model_file_prefix = 'result/' + MODE + '_model_'
lc_results = []
for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_accuracy, train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_accuracy, valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    epoch_result = dict()
    epoch_result['epoch'] = epoch
    epoch_result['train_loss'] = train_loss
    epoch_result['train_accuracy'] = train_accuracy
    epoch_result['valid_loss'] = valid_loss
    epoch_result['valid_accuracy'] = valid_accuracy
    lc_results.append(epoch_result)

    if len(sorted_results) < top_k or valid_accuracy > best_valid_accuracy:
        model_file = ''
        if len(sorted_results) >= top_k:
            model_file = sorted_results[-1][2]
            sorted_results.pop()
        else:
            model_file = model_file_prefix + str(available_model_id) + '.pt'
            available_model_id += 1

        item = (valid_accuracy, epoch_result, model_file)
        sorted_results.append(item)
        sorted_results = sorted(sorted_results, key=lambda x: x[0], reverse=True)
        best_valid_accuracy = sorted_results[-1][1]['valid_accuracy']
        torch.save(model.state_dict(), model_file)

    
    # if valid_accuracy > best_valid_accuracy:
    #     best_valid_accuracy = valid_accuracy
    #     torch.save(model.state_dict(), model_file)
    #     best_epoch = epoch
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'top-model.pt')
    if (epoch + 1) % report_iter == 0:
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Accuracy: {valid_accuracy:.2f}%')

epoch_result_list = [sorted_results[idx][1] for idx in range(len(sorted_results))]
train_df = pd.DataFrame(epoch_result_list)
train_path = 'result/' + MODE + '_train.csv'
train_df.to_csv(train_path, index=False)

lc_path = 'result/' + MODE + '_lc.csv'
lc_df = pd.DataFrame(lc_results)
lc_df.to_csv(lc_path, index=False)

print(train_df)


# print(f'Best Epoch: {best_epoch} |  Val. Accuracy: {best_valid_accuracy:.2f}%')


# Finally, we load the parameters from our best validation loss and get our results on the test set.
# 
# We get the improved test perplexity whilst almost being twice as fast!

# In[196]:

test_loss_list = []
test_accuracy_list = []
for idx in range(top_k):
    model_file = model_file_prefix + str(idx) + '.pt'
    model.load_state_dict(torch.load(model_file, map_location=device))
    test_accuracy, test_loss = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} |  Test Accuracy: {test_accuracy:.2f}%')
    test_loss_list.append(test_loss)
    test_accuracy_list.append(test_accuracy)

avg_test_loss = sum(test_loss_list) / len(test_loss_list)
avg_test_accuracy = sum(test_accuracy_list) / len(test_accuracy_list)
min_test_loss = min(test_loss_list)
max_test_accuracy = max(test_accuracy_list)
print(f'Avg Test Loss: {avg_test_loss:.3f} |  Avg Test Accuracy: {avg_test_accuracy:.2f}%')
print(f'Min Test Loss: {min_test_loss:.3f} |  Max Test Accuracy: {max_test_accuracy:.2f}%')

test_result = pd.DataFrame([{'avg_loss': avg_test_loss, 'avg_accuracy': avg_test_accuracy, 'min_loss': min_test_loss,
                              'max_accuracy': max_test_accuracy}])
test_path = 'result/' + MODE + '_test.csv'
test_result.to_csv(test_path, index=False)