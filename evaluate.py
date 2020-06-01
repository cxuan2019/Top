#!/usr/bin/env python3

from itertools import zip_longest
from data.tree import Tree
from typing import Counter, Dict, Optional
import argparse
import pandas as pd


class Calculator:
    def __init__(self, strict: bool = False) -> None:
        self.num_gold_nt: int = 0
        self.num_pred_nt: int = 0
        self.num_matching_nt: int = 0
        self.strict: bool = strict

    def get_metrics(self):
        precision: float = (
            self.num_matching_nt / self.num_pred_nt) if self.num_pred_nt else 0
        recall: float = (
            self.num_matching_nt / self.num_gold_nt) if self.num_gold_nt else 0
        f1: float = (2.0 * precision * recall /
                     (precision + recall)) if precision + recall else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def add_instance(self, gold_tree: Tree,
                     pred_tree: Optional[Tree] = None) -> None:
        node_info_gold: Counter = self._get_node_info(gold_tree)
        self.num_gold_nt += sum(node_info_gold.values())

        if pred_tree:
            node_info_pred: Counter = self._get_node_info(pred_tree)
            self.num_pred_nt += sum(node_info_pred.values())
            self.num_matching_nt += sum(
                (node_info_gold & node_info_pred).values())

    def _get_node_info(self, tree) -> Counter:
        nodes = tree.root.list_nonterminals()
        node_info: Counter = Counter()
        for node in nodes:
            node_info[(node.label, self._get_span(node))] += 1
        return node_info

    def _get_span(self, node):
        return node.get_flat_str_spans(
        ) if self.strict else node.get_token_span()

terminal_list = []
def is_non_terminal(word):
    if not word:
        return False
    if word.startswith('[IN:') or word.startswith('[SL:') or word == ']':
        return True

    if word not in terminal_list:
        terminal_list.append((word))
    return False


def comp_gold_and_pred(gold, pred) -> (int, int):
    gold_list = gold.split()
    pred_list = pred.split()

    gold_size = len(gold_list)
    pred_size = len(pred_list)
    non_terminal_match = 1
    terminal_match = 1

    gold_non_terminal_list = []
    gold_terminal_list = []
    current_span = []
    for i in range(0, gold_size):
        word = gold_list[i]
        if is_non_terminal(word):
            gold_non_terminal_list.append(word)
            if len(current_span) > 0:
                gold_terminal_list.append(" ".join(current_span))
                current_span = []
        else:
            current_span.append(word)
    if len(current_span) > 0:
        gold_terminal_list.append(" ".join(current_span))

    pred_non_terminal_list = []
    pred_terminal_list = []
    current_span = []
    for i in range(0, pred_size):
        word = pred_list[i]
        if is_non_terminal(word):
            pred_non_terminal_list.append(word)
            if len(current_span) > 0:
                pred_terminal_list.append(" ".join(current_span))
                current_span = []
        else:
            current_span.append(word)
    if len(current_span) > 0:
        pred_terminal_list.append(" ".join(current_span))

    # print("Non-terminal: {} || {}".format(gold_non_terminal_list, pred_non_terminal_list))
    # print("Terminal: {} || {}".format(gold_terminal_list, pred_terminal_list))
    # print("")

    return (gold_non_terminal_list == pred_non_terminal_list, gold_terminal_list == pred_terminal_list)



# def evaluate_predictions(gold_filename: str, pred_filename: str) -> Dict:
def evaluate_predictions(gold_filename: str, mode: str) -> Dict:

    instance_count: int = 0
    exact_matches: int = 0
    non_terminal_matches: int = 0
    terminal_matches: int = 0
    invalid_preds: float = 0
    labeled_bracketing_scores = Calculator(strict=False)
    tree_labeled_bracketing_scores = Calculator(strict=True)

    gold_list = []
    pred_list = []
    exact_match_list = []
    tree_depth_list = []
    tree_size_list = []

    with open(gold_filename) as gold_file:
        # for gold_line, pred_line in zip_longest(gold_file, pred_file):
        for gold_line in gold_file:

            try:
                gold_lines = gold_line.strip().split("\t")
                gold_line = gold_lines[1]
                pred_line = gold_lines[2] #pred_line.strip()
            except AttributeError:
                print("WARNING: check format and length of files")
                quit()

            (non_terminal_match, terminal_match) = comp_gold_and_pred(gold_line, pred_line)
            non_terminal_matches += non_terminal_match
            terminal_matches += terminal_match

            try:
                gold_tree = Tree(gold_line)
                instance_count += 1
            except ValueError:
                print("FATAL: found invalid line in gold file:", gold_line)
                quit()

            try:
                pred_tree = Tree(pred_line)
                labeled_bracketing_scores.add_instance(gold_tree, pred_tree)
                tree_labeled_bracketing_scores.add_instance(
                    gold_tree, pred_tree)
            except ValueError:
                # print("WARNING: found invalid line in pred file:", pred_line)
                invalid_preds += 1
                # (non_terminal_match, terminal_match) = comp_gold_and_pred(gold_line, pred_line)
                # non_terminal_matches += non_terminal_match
                # terminal_matches += terminal_match
                labeled_bracketing_scores.add_instance(gold_tree)
                tree_labeled_bracketing_scores.add_instance(gold_tree)
                continue

            if str(gold_tree) == str(pred_tree):
                exact_matches += 1
                exact_match_list.append(1)
            else:
                exact_match_list.append(0)

            if mode == 'sketch':
                tree_depth_list.append(gold_tree.root.get_depth()-1)
            else:
                tree_depth_list.append(gold_tree.root.get_depth()-2)
            tree_size_list.append(gold_tree.root.get_size())
            gold_list.append(gold_line)
            pred_list.append(pred_line)
            # if gold_tree.get_compact_form(sketch=True) == pred_tree.get_compact_form(sketch=True):
            #     non_terminal_matches += 1
            #
            # if gold_tree.root.get_str_token_spans() == pred_tree.root.get_str_token_spans():
            #     terminal_matches += 1

    result_file = mode + '.final_result.tsv'
    result = {'gold': gold_list, 'pred':pred_line, 'match':exact_match_list, 'size': tree_size_list, 'depth': tree_depth_list}
    result_df = pd.DataFrame.from_dict(result)
    result_df.to_csv(result_file, sep='\t', header=True, index=False)

    exact_match_fraction: float = (
        exact_matches / instance_count) if instance_count else 0
    tree_validity_fraction: float = (
        1 - (invalid_preds / instance_count)) if instance_count else 0

    # print("terminal list: {}".format(terminal_list))
    return {
        "instance_count":
        instance_count,
        "invalid_preds":
        invalid_preds,
        "exact_match_count":
        exact_matches,
        "exact_match_accu":
        exact_match_fraction,
        "non_terminal_match_Error_count":
        (instance_count-non_terminal_matches),
        "terminal_match_Error_count":
        (instance_count - terminal_matches),
        "total_Error_count":
        (instance_count - exact_matches)
        # "labeled_bracketing_scores":
        # labeled_bracketing_scores.get_metrics(),
        # "tree_labeled_bracketing_scores":
        # tree_labeled_bracketing_scores.get_metrics(),
        # "tree_validity":
        # tree_validity_fraction
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate TOP-representation predictions')
    parser.add_argument(
        'gold',
        type=str,
        help='file with each row in format: utterance <tab> tokenized_utterance <tab> TOP-representation'
    )
    parser.add_argument(
        'mode',
        type=str,
        help='lotv|compact|compact_copy|sketch|compact_further')
    args = parser.parse_args()

    # print(evaluate_predictions(args.gold, args.pred))
    print(evaluate_predictions(args.gold, args.mode))

