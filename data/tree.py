#!/usr/bin/env python3

from typing import List, Optional, Tuple

BRACKET_OPEN = "["
BRACKET_CLOSE = "]"
PREFIX_INTENT = "IN:"
PREFIX_SLOT = "SL:"


class Node:
    """
    A generalization of Root / Intent / Slot / Token
    """
    def __init__(self, label: str) -> None:
        self.label: str = label
        self.children: List[Node] = []
        self.parent: Optional[Node] = None

    def validate_node(self) -> None:
        for child in self.children:
            child.validate_node()

    def list_nonterminals(self):
        non_terminals: List[Node] = []
        for child in self.children:
            if type(child) != Root and type(child) != Token:
                non_terminals.append(child)
                non_terminals += child.list_nonterminals()
        return non_terminals

    def get_size(self) -> int:
        return len(self.list_nonterminals())

    def get_depth(self) -> int:
        if len(self.children) == 0:
            return 1
        child_depths = []
        for child in self.children:
            child_depths.append(child.get_depth())
        return max(child_depths) + 1

    def get_token_indices(self) -> List[int]:
        indices: List[int] = []
        if self.children:
            for child in self.children:
                if type(child) == Token:
                    indices.append(child.index)
                else:
                    indices += child.get_token_indices()
        return indices

    def get_token_span(self) -> Optional[Tuple[int, int]]:
        indices = self.get_token_indices()
        if indices:
            return (min(indices), max(indices) + 1)
        return None

    def get_flat_str_spans(self) -> str:
        str_span: str = str(self.get_token_span()) + ": "
        if self.children:
            for child in self.children:
                str_span += str(child)
        return str_span

    def get_str_token_spans(self) -> List[str]:
        token_spans = []
        if len(self.children) == 0:
            if type(self) == Token:
                token_spans.append(self.label)
            return token_spans

        current_span = []
        for child in self.children:
            if type(child) == Token:
                current_span.extend(child.get_str_token_spans())
            else:
                if len(current_span) > 0:
                    token_spans.append(" ".join(current_span))
                    current_span = []
                token_spans.extend(child.get_str_token_spans())
        if len(current_span) > 0:
            token_spans.append(" ".join(current_span))
        return token_spans

    def get_compact_form(self, sketech) -> str:
        str_repr: str = ""
        if type(self) == Token:
            return str_repr

        if type(self) == Intent or type(self) == Slot:
            str_repr = BRACKET_OPEN
        if type(self) != Root:
            str_repr += str(self.label) + " "

        has_nonterminal_child = len(self.list_nonterminals()) > 0
        if not has_nonterminal_child:
            if type(self) == Slot and not sketech:
                span = self.get_token_span()
                # str_repr += str(span[0]) + " " + str(span[1]) + " "  # str(span[0]) + " " + str(span[1] - 1) + " "
                if (span[1] - span[0]) == 1:
                    str_repr += str(span[0]) + " "
                else:
                    str_repr += str(span[0]) + " " + str(span[1]) + " "
            # else:
            #     print("Warning: intent {} has no noterminal: ({})".format(self.label, str(self)))
        if self.children:
            for child in self.children:
                str_repr += child.get_compact_form(sketech)
        if type(self) == Intent or type(self) == Slot:
            str_repr += BRACKET_CLOSE + " "
        return str_repr

    def __repr__(self) -> str:
        str_repr: str = ""
        if type(self) == Intent or type(self) == Slot:
            str_repr = BRACKET_OPEN
        if type(self) != Root:
            str_repr += str(self.label) + " "
        if self.children:
            for child in self.children:
                str_repr += str(child)
        if type(self) == Intent or type(self) == Slot:
            str_repr += BRACKET_CLOSE + " "
        return str_repr


class Root(Node):
    def __init__(self) -> None:
        super().__init__("ROOT")

    def validate_node(self) -> None:
        super().validate_node()
        for child in self.children:
            if type(child) == Slot or type(child) == Root:
                raise TypeError(
                    "A Root's child must be an Intent or Token: " + self.label)
            elif self.parent is not None:
                raise TypeError(
                    "A Root should not have a parent: " + self.label)


class Intent(Node):
    def __init__(self, label: str) -> None:
        super().__init__(label)

    def validate_node(self) -> None:
        super().validate_node()
        for child in self.children:
            if type(child) == Intent or type(child) == Root:
                raise TypeError(
                    "An Intent's child must be a slot or token: " + self.label)


class Slot(Node):
    def __init__(self, label: str) -> None:
        super().__init__(label)

    def validate_node(self) -> None:
        super().validate_node()
        for child in self.children:
            if type(child) == Slot or type(child) == Root:
                raise TypeError("An Slot's child must be an intent or token: "
                                + self.label)


class Token(Node):
    def __init__(self, label: str, index: int) -> None:
        super().__init__(label)
        self.index: int = index

    def validate_node(self) -> None:
        if len(self.children) > 0:
            raise TypeError("A Token {} can't have children: {}".format(
                self.label, str(self.children)))


class Tree:
    def __init__(self, top_repr: str) -> None:
        self.root = Tree.build_tree(top_repr)
        try:
            self.validate_tree()
        except ValueError as v:
            raise ValueError("Tree validation failed: {}".format(v))

    @staticmethod
    def build_tree(top_repr: str) -> Root:
        root = Root()
        node_stack: List[Node] = [root]
        token_count: int = 0

        for item in top_repr.split():
            if item == BRACKET_CLOSE:
                if not node_stack:
                    raise ValueError("Tree validation failed")
                node_stack.pop()

            elif item.startswith(BRACKET_OPEN):
                label: str = item[1:]
                if label.startswith(PREFIX_INTENT):
                    node_stack.append(Intent(label))
                elif label.startswith(PREFIX_SLOT):
                    node_stack.append(Slot(label))
                else:
                    raise NameError(
                        "Nonterminal label {} must start with {} or {}".format(
                            label, PREFIX_INTENT, PREFIX_SLOT))

                if len(node_stack) < 2:
                    raise ValueError("Tree validation failed")
                node_stack[-1].parent = node_stack[-2]
                node_stack[-2].children.append(node_stack[-1])

            else:
                token = Token(item, token_count)
                token_count += 1
                if not node_stack:
                    raise ValueError("Tree validation failed")
                token.parent = node_stack[-1]
                node_stack[-1].children.append(token)

        if len(node_stack) > 1:
            raise ValueError("Tree validation failed")

        return root

    def validate_tree(self) -> None:
        try:
            self.root.validate_node()
            for child in self.root.children:
                child.validate_node()
        except TypeError as t:
            raise ValueError("Failed validation for {} \n {}".format(
                self.root, str(t)))

    def get_compact_form(self, sketch=False) -> str:
        return self.root.get_compact_form(sketch).strip()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.root == other.root

    def __repr__(self) -> str:
        return repr(self.root).strip()


def main():
    # test = '[IN:GET_ESTIMATED_ARRIVAL 0 0 0 0 0 [SL:DESTINATION [IN:GET_LOCATION [SL:CATEGORY_LOCATION 0 0 ] ] ] [SL:DATE_TIME_ARRIVAL 0 0 ] 0 0 0 [SL:DATE_TIME_DEPARTURE 0 ] ]'
    test = "[IN:GET_EVENT 0 [SL:LOCATION [IN:GET_LOCATION [SL:SEARCH_RADIUS 0 ] [SL:LOCATION_USER 0 ] ] ] ]"
    tree = Tree(test)
    # print(tree.root.get_token_span())
    # print(tree.root.get_flat_str_spans())
    print(tree)
    print("tree depth: {}; size: {}".format(tree.root.get_depth()-2, tree.root.get_size()))
    print("sketch: {}".format(tree.root.get_compact_form(sketech=True)))
    print("Token Spans: {}".format(tree.root.get_str_token_spans()))

    print("")
    test = "[IN:GET_EVENT [SL:LOCATION [IN:GET_LOCATION [SL:SEARCH_RADIUS 1 2 ] [SL:LOCATION_USER 2 3 ] ] ] ]"
    tree = Tree(test)
    # print(tree.root.get_token_span())
    # print(tree.root.get_flat_str_spans())
    print(tree)
    print("tree depth: {}; size: {}".format(tree.root.get_depth()-2, tree.root.get_size()))
    print("sketch: {}".format(tree.root.get_compact_form(sketech=True)))
    print("Token Spans: {}".format(tree.root.get_str_token_spans()))

    print("")
    test = "[IN:GET_EVENT [SL:LOCATION [IN:GET_LOCATION [SL:SEARCH_RADIUS ] [SL:LOCATION_USER ] ] ] ]"
    tree = Tree(test)
    # print(tree.root.get_token_span())
    # print(tree.root.get_flat_str_spans())
    print(tree)
    print("tree depth: {}; size: {}".format(tree.root.get_depth()-2, tree.root.get_size()))
    print("sketch: {}".format(tree.root.get_compact_form(sketech=True)))
    print("Token Spans: {}".format(tree.root.get_str_token_spans()))

    # print(tree.get_compact_form())

if __name__ == "__main__":
    main()