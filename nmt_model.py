#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from collections import namedtuple
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from model_embeddings import ModelEmbeddings
from vocab import Vocab

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """

    def __init__(self, src_embed_size, dst_embed_size, hidden_size, vocab, dropout_rate=0.2, use_pos_embed=False,
                 use_copy=False, max_src_len=48):
        """ Init NMT Model.

        @param src_embed_size (int): encoder Embedding size (dimensionality)
        @param dst_embed_size (int): decoder Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for  documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(src_embed_size, dst_embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.use_pos_embed = use_pos_embed
        self.use_copy = use_copy
        self.max_src_len = max_src_len

        # default values
        self.encoder = None
        self.decoder = None
        self.h_projection = None
        self.c_projection = None
        self.att_projection = None
        self.combined_output_projection = None
        self.target_vocab_projection = None
        self.dropout = None

        ### YOUR CODE HERE (~8 Lines)
        ### TODO - Initialize the following variables:
        ###     self.encoder (Bidirectional LSTM with bias)
        ###     self.decoder (LSTM Cell with bias)
        ###     self.h_projection (Linear Layer with no bias), called W_{h} in the PDF.
        ###     self.c_projection (Linear Layer with no bias), called W_{c} in the PDF.
        ###     self.att_projection (Linear Layer with no bias), called W_{attProj} in the PDF.
        ###     self.combined_output_projection (Linear Layer with no bias), called W_{u} in the PDF.
        ###     self.target_vocab_projection (Linear Layer with no bias), called W_{vocab} in the PDF.
        ###     self.dropout (Dropout Layer)
        ###
        ### Use the following docs to properly initialize these variables:
        ###     LSTM:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        ###     LSTM Cell:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell
        ###     Linear Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        ###     Dropout Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout
        self.encoder = nn.LSTM(src_embed_size, hidden_size, num_layers=1, bias=True, bidirectional=True)
        self.decoder = nn.LSTMCell(dst_embed_size+hidden_size, hidden_size, bias=True)
        self.h_projection = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.c_projection = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.att_projection = nn.Linear(2*hidden_size, hidden_size, bias=False) if not use_pos_embed else \
            nn.Linear(2 * hidden_size+dst_embed_size, hidden_size, bias=False)
        if use_pos_embed:
            self.combined_output_projection = nn.Linear(3 * hidden_size+dst_embed_size, hidden_size, bias=False)
        elif use_copy:
            self.combined_output_projection = nn.Linear(3 * hidden_size + max_src_len, hidden_size, bias=False)
        else:
            self.combined_output_projection = nn.Linear(3*hidden_size, hidden_size, bias=False)
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.tgt), bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        # self.device = torch.device('cpu')


        ### END YOUR CODE

    def forward(self, source: torch.Tensor, source_lengths: List[int], target: torch.Tensor) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (Tensor): padded source sentences with shape (src_len, b)
        @param source_lengths (List[int]): list of source sentence lengths
        @param target (Tensor): padded target sentences with shape (src_len, b)

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # # Compute sentence lengths
        # source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        # source_padded = self.vocab.src.to_input_tensor(source, device=self.device)  # Tensor: (src_len, b)
        # target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)  # Tensor: (tgt_len, b)
        source_padded = source
        target_padded = target
        src_len = source.size(0)
        batch_size = source.size(1)


        ###     Run the network forward:
        ###     1. Apply the encoder to `source_padded` by calling `self.encode()`
        ###     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        ###     3. Apply the decoder to compute combined-output by calling `self.decode()`
        ###     4. Compute log probability distribution over the target vocabulary using the
        ###        combined_outputs returned by the `self.decode()` function.

        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        # target_masks = (target_padded != self.vocab.tgt['<pad>']).float() #dst_pad_token_idx
        target_masks = (target_padded != self.vocab.dst_pad_token_idx).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores


    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that 
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """
        enc_hiddens, dec_init_state = None, None

        ### YOUR CODE HERE (~ 8 Lines)
        ### TODO:
        ###     1. Construct Tensor `X` of source sentences with shape (src_len, b, e) using the source model embeddings.
        ###         src_len = maximum source sentence length, b = batch size, e = embedding size. Note
        ###         that there is no initial hidden state or cell for the decoder.
        ###     2. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.
        ###         - Before you can apply the encoder, you need to apply the `pack_padded_sequence` function to X.
        ###         - After you apply the encoder, you need to apply the `pad_packed_sequence` function to enc_hiddens.
        ###         - Note that the shape of the tensor returned by the encoder is (src_len b, h*2) and we want to
        ###           return a tensor of shape (b, src_len, h*2) as `enc_hiddens`.
        ###     3. Compute `dec_init_state` = (init_decoder_hidden, init_decoder_cell):
        ###         - `init_decoder_hidden`:
        ###             `last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the h_projection layer to this in order to compute init_decoder_hidden.
        ###             This is h_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###         - `init_decoder_cell`:
        ###             `last_cell` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the c_projection layer to this in order to compute init_decoder_cell.
        ###             This is c_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###
        ### See the following docs, as you may need to use some of the following functions in your implementation:
        ###     Pack the padded sequence X before passing to the encoder:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence
        ###     Pad the packed sequence, enc_hiddens, returned by the encoder:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_packed_sequence
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tensor Permute:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute
        #convert sentences to indexes
        # source_indexes = None
        # if torch.cuda.is_available():
        #     source_indexes = torch.cuda.LongTensor(self.vocab.src.words2indices(source_padded.tolist())).to(self.device)
        # else:
        #     source_indexes = torch.LongTensor(self.vocab.src.words2indices(source_padded.tolist())).to(self.device)
        # source_indexes = torch.LongTensor(self.vocab.src.words2indices(source_padded.tolist()))
        # X = self.model_embeddings.source(source_indexes)
        X = self.model_embeddings.source(source_padded)
        input = torch.nn.utils.rnn.pack_padded_sequence(X, source_lengths)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(input)
        enc_hiddens, _ = torch.nn.utils.rnn.pad_packed_sequence(enc_hiddens, batch_first=True)

        init_decoder_hidden = self.h_projection(torch.cat([last_hidden[0], last_hidden[1]], 1))
        init_decoder_cell = self.c_projection(torch.cat([last_cell[0], last_cell[1]], 1))
        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        ### END YOUR CODE

        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor, dec_init_state: Tuple[torch.Tensor, torch.Tensor],
               target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size.

        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        src_len = enc_hiddens.size(1)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        ### YOUR CODE HERE (~9 Lines)
        ### TODO:
        ###     1. Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`,
        ###         which should be shape (b, src_len, h),
        ###         where b = batch size, src_len = maximum source length, h = hidden size.
        ###         This is applying W_{attProj} to h^enc, as described in the PDF.
        ###     2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
        ###         where tgt_len = maximum target sentence length, b = batch size, e = embedding size.
        ###     3. Use the torch.split function to iterate over the time dimension of Y.
        ###         Within the loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
        ###             - Squeeze Y_t into a tensor of dimension (b, e). 
        ###             - Construct Ybar_t by concatenating Y_t with o_prev.
        ###             - Use the step function to compute the the Decoder's next (cell, state) values
        ###               as well as the new combined output o_t.
        ###             - Append o_t to combined_outputs
        ###             - Update o_prev to the new o_t.
        ###     4. Use torch.stack to convert combined_outputs from a list length tgt_len of
        ###         tensors shape (b, h), to a single tensor shape (tgt_len, b, h)
        ###         where tgt_len = maximum target sentence length, b = batch size, h = hidden size.
        ###
        ### Note:
        ###    - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###      over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###   
        ### Use the following docs to implement this functionality:
        ###     Zeros Tensor:
        ###         https://pytorch.org/docs/stable/torch.html#torch.zeros
        ###     Tensor Splitting (iteration):
        ###         https://pytorch.org/docs/stable/torch.html#torch.split
        ###     Tensor Dimension Squeezing:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tensor Stacking:
        ###         https://pytorch.org/docs/stable/torch.html#torch.stack
        # enc_hiddens_proj = None
        combined_inputs = enc_hiddens
        if self.use_pos_embed:
            input_pos_embed = self.create_input_pos_embed(src_len, batch_size)
            combined_inputs = torch.cat((enc_hiddens, input_pos_embed), dim=2)
            enc_hiddens_proj = self.att_projection(combined_inputs)
        else:
            enc_hiddens_proj = self.att_projection(enc_hiddens)

        # print("Device is: {}".format(self.device))
        # target_indexes = None
        # if torch.cuda.is_available():
        #     target_indexes = torch.cuda.LongTensor(self.vocab.tgt.words2indices(target_padded.tolist())).to(self.device)
        # else:
        #     target_indexes = torch.LongTensor(self.vocab.tgt.words2indices(target_padded.tolist())).to(self.device)
        # target_indexes = torch.LongTensor(self.vocab.tgt.words2indices(target_padded.tolist()))
        # Y = self.model_embeddings.target(target_indexes)
        Y = self.model_embeddings.target(target_padded)

        for Y_t in torch.split(Y, 1, dim=0):
            Y_t = torch.squeeze(Y_t, dim=0)
            Ybar_t = torch.cat([Y_t, o_prev], dim=1)
            dec_state, o_t, _ = self.step(Ybar_t, dec_state, combined_inputs, enc_hiddens_proj, enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t

        combined_outputs = torch.stack(combined_outputs, dim=0)
        ### END YOUR CODE

        return combined_outputs

    def step(self, Ybar_t: torch.Tensor,
             dec_state: Tuple[torch.Tensor, torch.Tensor],
             enc_hiddens: torch.Tensor,
             enc_hiddens_proj: torch.Tensor,
             enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor (+ input_pos_embed if use_pos_embed), with shape (b, src_len, h * 2 (+ dsb_embed_size)), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length. 

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        combined_output = None

        ### YOUR CODE HERE (~3 Lines)
        ### TODO:
        ###     1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        ###     2. Split dec_state into its two parts (dec_hidden, dec_cell)
        ###     3. Compute the attention scores e_t, a Tensor shape (b, src_len). 
        ###        Note: b = batch_size, src_len = maximum source length, h = hidden size.
        ###
        ###       Hints:
        ###         - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        ###         - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        ###         - Use batched matrix multiplication (torch.bmm) to compute e_t.
        ###         - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
        ###         - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###             over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        ### Use the following docs to implement this functionality:
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor Unsqueeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        ###     Tensor Squeeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        dec_hidden, dec_cell = self.decoder(Ybar_t, dec_state)
        dec_state = (dec_hidden, dec_cell)
        e_t = torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden, 2))
        e_t = torch.squeeze(e_t, 2)
        ### END YOUR CODE

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        ### YOUR CODE HERE (~6 Lines)
        ### TODO:
        ###     1. Apply softmax to e_t to yield alpha_t
        ###     2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
        ###         attention output vector, a_t.
        # $$     Hints:
        ###           - alpha_t is shape (b, src_len)
        ###           - enc_hiddens is shape (b, src_len, 2h)
        ###           - a_t should be shape (b, 2h)
        ###           - You will need to do some squeezing and unsqueezing.
        ###     Note: b = batch size, src_len = maximum source length, h = hidden size.
        ###
        ###     3. Concatenate a_t with dec_hidden compute tensor U_t
        ###     4. Apply the combined output projection layer to U_t to compute tensor V_t
        ###     5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
        ###
        ### Use the following docs to implement this functionality:
        ###     Softmax:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softmax
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor View:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tanh:
        ###         https://pytorch.org/docs/stable/torch.html#torch.tanh
        soft_max = torch.nn.Softmax(1)
        alpha_t = soft_max(e_t)

        a_t = torch.bmm(torch.unsqueeze(alpha_t, 1), enc_hiddens)
        a_t = torch.squeeze(a_t, 1)

        # U_t = torch.cat([a_t, dec_hidden], 1)
        if self.use_copy:
            U_t = torch.cat([dec_hidden, a_t], 1)
            src_len = alpha_t.size(1)
            if src_len == self.max_src_len:
                alpha_t_ext = alpha_t
            else:
                batch_size = alpha_t.size(0)
                ext = torch.zeros(batch_size, (self.max_src_len - src_len)).to(self.device)
                alpha_t_ext = torch.cat([alpha_t, ext], 1)
            U_t = torch.cat([dec_hidden, a_t, alpha_t_ext], 1) # attention based copy
        else:
            U_t = torch.cat([dec_hidden, a_t], 1)

        V_t = self.combined_output_projection(U_t)

        O_t = self.dropout(torch.tanh(V_t))
        ### END YOUR CODE

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size. 
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.
        
        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def beam_search(self, src_sent: torch.Tensor, src_sent_len: int, beam_size: int = 5, max_decoding_time_step: int = 30) -> List[
        Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (Tensor): a single source sentence (words) with shape (sentence len, )
        @param src_sent_len int; src_sent length: int
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        # src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)
        src_sents_var = torch.unsqueeze(src_sent, 1)

        # src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings, dec_init_vec = self.encode(src_sents_var, [src_sent_len])

        # src_encodings_att_linear = None
        combined_inputs = src_encodings
        if self.use_pos_embed:
            input_pos_embed = self.create_input_pos_embed(src_sent_len, 1)
            combined_inputs = torch.cat((src_encodings, input_pos_embed), dim=2)
            src_encodings_att_linear = self.att_projection(combined_inputs)
        else:
            src_encodings_att_linear = self.att_projection(src_encodings)


        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        # eos_id = self.vocab.tgt['</s>']
        eos_id = self.vocab.dst_eos_token_idx

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = combined_inputs.expand(hyp_num,
                                                     combined_inputs.size(1),
                                                     combined_inputs.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt.stoi[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _ = self.step(x, h_tm1,
                                                exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.itos[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str, use_pos_embed: bool, use_copy: bool):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], use_pos_embed=use_pos_embed, use_copy=use_copy, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(src_embed_size=self.model_embeddings.src_embed_size, dst_embed_size=self.model_embeddings.dst_embed_size,
                         hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

    def load_pretrained_embeddings(self, vocab: Vocab):
        self.model_embeddings.source.weight.data.copy_(vocab.src.vectors)

    def create_input_pos_embed(self, src_len, batch_size):
        pos_list = [[self.vocab.tgt.stoi[str(i)] for i in range(0, src_len)]]
        input_positions = torch.LongTensor(pos_list).to(self.device)  # [1, src_len]
        input_positions = input_positions.repeat(batch_size, 1)  # [ batch_size, src_len]
        input_embed = self.model_embeddings.target(input_positions)
        return input_embed # [ batch_size, src_len, dst_embed_size]
