import os.path

import numpy as np
import pandas as pd

from data.tree import Tree


data_dir = '/Users/cxuan/Projects/SCPD_XCS224U/project/top/data/'
target_list = ['train', 'eval', 'test']
intent_list = []
slot_list = []
vocab_list = []

def generate_y_compact(y):
    y = str(y)
    if not y:
        return y

    return Tree(y).get_compact_form()


def generate_y_sketch(y):
    y = str(y)
    if not y:
        return y

    return Tree(y).get_compact_form(True)


def generate_y_lotv(y):
    y = str(y)
    if not y:
        return y

    ret = []
    for word in y.split(' '):
        word = word.strip()
        if word == ']':
            ret.append(word)
            continue
        sub_word = word[1:]
        if (sub_word in intent_list) or (sub_word in slot_list):
            ret.append(word)
        else:
            ret.append('0')

    return ' '.join(ret)


def get_intents_slots(y):
    y = str(y)
    if not y:
        return

    for word in y.split(sep=' '):
        word = word.strip()
        if word.startswith('[IN:'):
            intent = word[1:]
            if intent not in intent_list:
                intent_list.append(intent)
        elif word.startswith('[SL:'):
            slot = word[1:]
            if slot not in slot_list:
                slot_list.append(slot)


def process_target_for_mix(target):
    target_path = data_dir + target + '_compact.tsv'

    if not os.path.exists(target_path):
        print("file {} not exists!".format(target_path))
        return

    col = ['X', 'Y']
    df = pd.read_csv(target_path, sep='\t', names=col)

    def mix_word_and_pos(words):
        word_list = words.split()
        mix_list = [str(idx) + ' ' + word_list[idx] for idx in range(0, len(word_list))]
        return ' '.join(mix_list).strip()

    df['X_MIX'] = df['X'].apply(mix_word_and_pos)
    out_path = data_dir + target + '_mix.tsv'
    df[['X_MIX', 'Y']].to_csv(out_path, sep='\t', index=False, header=False)


def get_stats():
    for form in ['lotv', 'compact', 'compact_further', 'sketch']:
        target_path = data_dir + 'train_' + form + '.tsv'
        if not os.path.exists(target_path):
            print("file {} not exists!".format(target_path))
            return
        col = ['X', 'Y']
        df = pd.read_csv(target_path, sep='\t', names=col)
        col = 'Y'
        # if form == 'token':
        #     col = 'X'
        col_len = col + '_LEN'
        #
        # def get_nonterminal_len(x):
        #     return len(Tree(str(x)).root.list_nonterminals())
        #
        # vocab_list = []
        intent_list = []
        slot_list = []

        def get_intents_slots(y):
            y = str(y)
            if not y:
                return

            for word in y.split(sep=' '):
                word = word.strip()
                if word.startswith('[IN:'):
                    intent = word[1:]
                    if intent not in intent_list:
                        intent_list.append(intent)
                elif word.startswith('[SL:'):
                    slot = word[1:]
                    if slot not in slot_list:
                        slot_list.append(slot)

        def get_vocabs(x):
            token_list = str(x).split()
            for token in token_list:
                if not token in vocab_list:
                    vocab_list.append(token)
            return 1

        # df[col_len] = df[col].apply(lambda x: len(str(x).split()))
        df[col_len] = df[col].apply(get_intents_slots)

        print(form)
        # print(len(vocab_list))
        # print(df[col_len].describe())
        print(len(intent_list))
        print(len(slot_list))
        print(" ")



def process_target(target):
    target_path = data_dir + target + '.tsv'

    if not os.path.exists(target_path):
        print("file {} not exists!".format(target_path))
        return

    col = ['raw', 'X', 'Y']
    df = pd.read_csv(target_path, sep='\t', names=col)

    # df['Y'].apply(get_intents_slots)
    # print("{}: {} valid records".format(target, len(df[~df['Y'].str.contains('IN:UNSUPPORTED')])))

    df = df[~df['Y'].str.contains('IN:UNSUPPORTED')]

    # For token
    # df_token = df[['X', 'Y']]
    # out_path = data_dir + target + '_token.tsv'
    # df_token.to_csv(out_path, sep='\t', index=False, header=False)

    # For LOTV
    # df['Y_LOTV'] = df['Y'].apply(generate_y_lotv)
    # df_lotv = df[['X', 'Y_LOTV']]
    # out_path = data_dir + target + '_lotv.tsv'
    # df_lotv.to_csv(out_path, sep='\t', index=False, header=False)
    #

    # For compact
    df['Y_COMPACT'] = df['Y'].apply(generate_y_compact)
    df_compact = df[['X', 'Y_COMPACT']]
    out_path = data_dir + target + '_compact.tsv'
    df_compact.to_csv(out_path, sep='\t', index=False, header=False)

    # For sketch
    # df['Y_SKETCH'] = df['Y'].apply(generate_y_sketch)
    # df_sketech = df[['X', 'Y_SKETCH']]
    # out_path = data_dir + target + '_sketch.tsv'
    # df_sketech.to_csv(out_path, sep='\t', index=False, header=False)




def main():
    get_stats()
    # for target in target_list:
    #     process_target(target)
        # process_target_for_mix(target)

    # print("Total {} intents and {} slots".format(len(intent_list), len(slot_list)))
    # print('')
    # print(intent_list)
    # print('')
    # print(slot_list)

if __name__ == "__main__":
    main()