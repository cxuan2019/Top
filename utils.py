#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math

import numpy as np
import spacy
from torchtext.data import Field, BucketIterator, TabularDataset


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    size = len(max(sents, key=len))
    for sent in sents:
        gap = size - len(sent)
        if gap > 0 :
            pad = [pad_token for i in range(gap)]
            sent.extend(pad)
        sents_padded.append(sent)

    return sents_padded


def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


def load_data(train_file, valid_file, test_file, device, BATCH_SIZE=128, use_pos_embed=False):
    spacy_en = spacy.load('en')

    def tokenize_trg(text):
        """
        Tokenizes parse]
        """
        return text.split()

    def tokenize_src(text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = Field(tokenize=tokenize_src,
                init_token='<s>',
                eos_token='</s>',
                lower=True,
                include_lengths=True)

    TRG = Field(tokenize=tokenize_trg,
                init_token='<s>',
                eos_token='</s>',
                lower=False)

    top_datafields = [('src', SRC), ('trg', TRG)]
    train_data, valid_data, test_data = TabularDataset.splits(path='data', train=train_file, validation=valid_file,
                                                              test=test_file, format='tsv', skip_header=False,
                                                              fields=top_datafields)
    SRC.build_vocab(train_data, vectors="glove.6B.200d")
    TRG.build_vocab(train_data, min_freq=1)

    # add position data to trg vocab
    MAX_SRC_LEN = 45
    for i in range(0, MAX_SRC_LEN):
        word = str(i)
        if word not in TRG.vocab.stoi:
            TRG.vocab.itos.append(word)
            TRG.vocab.stoi[word] = len(TRG.vocab.itos) - 1

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device)

    return SRC, TRG, train_iterator, valid_iterator, test_iterator

