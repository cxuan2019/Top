#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    run.py experiement --train-data=<file> --dev-data=<file> --test-data=<file> [options]
    run.py test --train-data=<file> --dev-data=<file> --test-data=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE


Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --use-pos-embed                         use word postion embedding
    --use-copy                              use attention-based copy
    --train-data=<file>                     train data file
    --dev-data=<file>                       dev data file
    --test-data=<file>                      test data file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --src-embed-size=<int>                  src embedding size [default: 200]
    --dst-embed-size=<int>                  dst embedding size [default: 128]
    --hidden-size=<int>                     hidden size [default: 512]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 3]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 2]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 30]
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 50]
    --input-feed                            use input feeding
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
"""
import math
import sys
import time
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.utils
import torch.nn as nn
from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu
from nmt_model import Hypothesis, NMT
from tqdm import tqdm
from torchtext.data import Field, BucketIterator, TabularDataset
from utils import read_corpus, batch_iter, load_data
from vocab import Vocab
import random


def evaluate_ppl(model, dev_iterator, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        # for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
        for i, batch in enumerate(dev_iterator):
            src_sents, src_sents_lens = batch.src
            tgt_sents = batch.trg
            loss = -model(src_sents, src_sents_lens, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def experiement(args: Dict, test_only, device):
    """ Train and Test the NMT Model.
    @param args (Dict): args from cmd line
    """
    # train_data_src = read_corpus(args['--train-src'], source='src')
    # train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')
    #
    # dev_data_src = read_corpus(args['--dev-src'], source='src')
    # dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')
    #
    # train_data = list(zip(train_data_src, train_data_tgt))
    # dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    use_pos_embed = False
    if args['--use-pos-embed']:
        use_pos_embed = True

    use_copy = False
    if args['--use-copy']:
        use_copy = True

    SRC, TRG, train_iterator, dev_iterator, test_iterator = load_data(args['--train-data'], args['--dev-data'],
                                                                      args['--test-data'], device, train_batch_size, (use_pos_embed or use_copy))

    vocab = Vocab(SRC, TRG)


    model = NMT(src_embed_size=int(args['--src-embed-size']),
                dst_embed_size=int(args['--dst-embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab,
                use_pos_embed=use_pos_embed,
                use_copy=use_copy)

    model.load_pretrained_embeddings(vocab)

    # print("args: {}".format(args))

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    # def init_weights(m):
    #     for name, param in m.named_parameters():
    #         if 'weight' in name:
    #             nn.init.normal_(param.data, mean=0, std=0.01)
    #         else:
    #             nn.init.constant_(param.data, 0)
    #
    # model.apply(init_weights)

    # vocab_mask = torch.ones(len(vocab.tgt))
    # vocab_mask[vocab.tgt['<pad>']] = 0

    print('use device: %s' % device, file=sys.stderr)
    print(model)

    para_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {para_count:,} trainable parameters')
    print("file path: {}".format(model_save_path))

    if test_only:
        model.eval()
        decode(args, test_iterator, vocab, device)
        exit(0)

    # perform training
    model.train()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        #perform training
        model.train()
        # for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
        for i, batch in enumerate(train_iterator):
            train_iter += 1

            optimizer.zero_grad()
            src_sents, src_sents_lens = batch.src
            tgt_sents = batch.trg
            batch_size = src_sents.shape[1]

            example_losses = -model(src_sents, src_sents_lens, tgt_sents)  # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

        # if train_iter % log_every == 0:
        # print("")
        print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
              'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter, report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

        train_time = time.time()
        report_loss = report_tgt_words = report_examples = 0.

        # perform validation
        # model.eval()
        # if train_iter % valid_niter == 0:
        # print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
        #                                                                                      cum_loss / cum_examples,
        #                                                                                      np.exp(cum_loss / cum_tgt_words),
        #                                                                                      cum_examples), file=sys.stderr)

        cum_loss = cum_examples = cum_tgt_words = 0.
        valid_num += 1

        # print('begin validation ...', file=sys.stderr)

        # compute dev. ppl and bleu
        dev_ppl = evaluate_ppl(model, dev_iterator, batch_size=128)  # dev batch size can be a bit larger
        valid_metric = -dev_ppl

        print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

        is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
        hist_valid_scores.append(valid_metric)

        if is_better:
            patience = 0
            # print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
            model.save(model_save_path)

            # also save the optimizers' state
            torch.save(optimizer.state_dict(), model_save_path + '.optim')
        elif patience < int(args['--patience']):
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

            if patience == int(args['--patience']):
                num_trial += 1
                print('hit #%d trial' % num_trial, file=sys.stderr)
                if num_trial == int(args['--max-num-trial']):
                    print('early stop!', file=sys.stderr)
                    # exit(0)
                    break

                # decay lr, and restore from previously best checkpoint
                lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                # load model
                params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                model.load_state_dict(params['state_dict'])
                model = model.to(device)

                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                # set new lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # reset patience
                patience = 0

        if epoch == int(args['--max-epoch']):
            print('reached maximum number of epochs!', file=sys.stderr)
            break

    # perform testing
    model.eval()
    decode(args, test_iterator, vocab, device)

def compute_match(gold: torch.Tensor, pre: List[int], eos_token_idx: int) -> float:
    """ Given decoding results and reference sentence, compute corpus-level accuracy score.
    @param gold (torch.Tensor): gold-standard reference target sentences, (tgt_len, )
    @param pre (List[int]): hypotheses for the reference
    @returns result (int): 0 or 1
    """
    # if references[0][0] == '<s>':
    #     references = [ref[1:-1] for ref in references]
    # bleu_score = corpus_bleu([[ref] for ref in references],
    #                          [hyp.value for hyp in hypotheses])

    gold = gold.tolist()  # list[int]
    gold = gold[1:]
    valid_len = len(gold)
    if eos_token_idx in gold:
        valid_len = gold.index(eos_token_idx)
    if len(pre) < valid_len:
        return 0

    deltas = [abs(pre[i] - gold[i]) for i in range(valid_len)]
    return 1 if sum(deltas) == 0 else 0


def decode(args: Dict[str, str], test_iterator: BucketIterator, vocab: Vocab, device: torch.device):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    # print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    # test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    # if args['TEST_TARGET_FILE']:
    #     print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
    #     test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print("")
    print("Start Testing: load model from {}".format(args['--save-to']), file=sys.stderr)
    model = NMT.load(args['--save-to'], bool(args['--use-pos-embed']), bool(args['--use-copy']))

    if args['--cuda']:
        # model = model.to(torch.device("cuda:0"))
        model = model.to(device)

    beam_size = int(args['--beam-size'])
    hypotheses = beam_search(model, test_iterator, beam_size,
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    thd = 3
    # for i in range(len(hypotheses)):
    #     if i >= thd:
    #         break
    #     print("Hypo {}:".format(i))
    #     for j in range(beam_size):
    #         print("   beam {}: {}".format(j, hypotheses[i][j]))

    # # Compute accuracy
    top_hypotheses = [hyps[0].value for hyps in hypotheses]
    match_count = 0
    src_list = []
    gold_list = []
    pre_list = []
    for i, batch in enumerate(test_iterator):
        src_sents, _ = batch.src
        src_sents = src_sents.permute(1, 0)
        trg_sents = batch.trg
        trg_sents = trg_sents.permute(1, 0)
        batch_size = trg_sents.size()[0]
        for j in range(batch_size):
            idx = i * batch_size + j
            pred = [vocab.tgt.stoi[token] for token in top_hypotheses[idx]]
            trg_sent = trg_sents[j]
            src_sent = src_sents[j]
            src = [vocab.src.itos[item] for item in src_sent.tolist()]
            gold = [vocab.tgt.itos[item] for item in trg_sent.tolist()]
            src = src[1:-1]
            gold = gold[1:gold.index('</s>')]
            src_list.append(" ".join(src))
            gold_list.append(" ".join(gold))
            pre_list.append(" ".join(top_hypotheses[idx]))
            if (idx < thd):
                print("ID: {}".format(idx))
                print("src: {}".format(" ".join(src)))
                print("gold: {}".format(" ".join(gold)))
                print("pre: {}".format(" ".join(top_hypotheses[idx])))

            match_count += compute_match(trg_sent, pred, vocab.dst_eos_token_idx)
    accuracy = match_count * 100 / len(top_hypotheses)
    print("Test Accuracy: {}".format(accuracy))
    result_file = args['--save-to'] + '.result.csv'
    result = {'hidden_size': [args['--hidden-size']], 'beam_size': [beam_size], 'accuracy': [accuracy]}
    result = pd.DataFrame.from_dict(result)
    result.to_csv(result_file)

    result_data_file = args['--save-to'] + '.result_data.tsv'
    result_data = {'src': src_list, 'gold': gold_list, 'pre': pre_list}
    result_df = pd.DataFrame.from_dict(result_data)
    result_df.to_csv(result_data_file, sep='\t', header=False, index=False)


def beam_search(model: NMT, test_iterator: BucketIterator, beam_size: int, max_decoding_time_step: int) -> List[
    List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_iterator BucketIterator: BucketIterator in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        # for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        for i, batch in enumerate(test_iterator):
            src_sents, src_sents_lens = batch.src
            src_sents = src_sents.permute(1, 0)
            for j in range(len(src_sents_lens)):
                src_sent = src_sents[j]
                example_hyps = model.beam_search(src_sent, src_sents_lens[j], beam_size=beam_size,
                                             max_decoding_time_step=max_decoding_time_step)
                hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check pytorch version
    assert (
                torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0 or above".format(
        torch.__version__)

    # seed the random number generators
    SEED = int(args['--seed'])  #2345 #

    random.seed(SEED)
    torch.manual_seed(SEED)
    if args['--cuda']:
        # torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)  # np.random.seed(seed * 13 // 7)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args['experiement']:
        experiement(args, False, device)
    elif args['test']:
        experiement(args, True, device)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
