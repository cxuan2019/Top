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


    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))


