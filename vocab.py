#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

import json
from collections import Counter
from itertools import chain
from typing import List

import torch
from docopt import docopt
from utils import read_corpus, pad_sents
import torchtext


class Vocab(object):
    """ Vocab encapsulating src and target langauges.
    """

    def __init__(self, SRC: torchtext.data.Field, TRG: torchtext.data.Field):
        """ Init Vocab.
        @param SRC (Field): field for source language
        @param TRG (Field): filed for target language
        """
        self.src = SRC.vocab
        self.tgt = TRG.vocab
        self.src_pad_token_idx = SRC.vocab.stoi[SRC.pad_token]
        self.dst_pad_token_idx = TRG.vocab.stoi[TRG.pad_token]
        self.dst_eos_token_idx = TRG.vocab.stoi['</s>']

    # @staticmethod
    # def build(src_sents, tgt_sents, vocab_size, freq_cutoff) -> 'Vocab':
    #     """ Build Vocabulary.
    #     @param src_sents (list[str]): Source sentences provided by read_corpus() function
    #     @param tgt_sents (list[str]): Target sentences provided by read_corpus() function
    #     @param vocab_size (int): Size of vocabulary for both source and target languages
    #     @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word.
    #     """
    #     assert len(src_sents) == len(tgt_sents)
    #
    #     print('initialize source vocabulary ..')
    #     src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff)
    #
    #     print('initialize target vocabulary ..')
    #     tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff)
    #
    #     return Vocab(src, tgt)
    #
    # def save(self, file_path):
    #     """ Save Vocab to file as JSON dump.
    #     @param file_path (str): file path to vocab file
    #     """
    #     json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), open(file_path, 'w'), indent=2)
    #
    # @staticmethod
    # def load(file_path):
    #     """ Load vocabulary from JSON dump.
    #     @param file_path (str): file path to vocab file
    #     @returns Vocab object loaded from JSON dump
    #     """
    #     entry = json.load(open(file_path, 'r'))
    #     src_word2id = entry['src_word2id']
    #     tgt_word2id = entry['tgt_word2id']
    #
    #     return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))


