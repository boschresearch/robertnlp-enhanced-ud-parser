# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Stefan Grünewald

from collections import defaultdict
from itertools import chain

from .label_index_matrix import LabelIndexMatrix


class DependencyAnnotatedSentence:
    """Class for representing a single sentence, annotated with dependencies between tokens.

    The dependencies within the sentence are represented as a matrix in which rows represent heads and columns represent
    dependents. A cell contains the relation holding between the head and the dependent, or a special symbol ([null]) in
    the case of no relation.

    The first token of a DependencyAnnotatedSentence should always be the special [root] token to ensure consistency
    between token indices and dependency matrix indices. This means that if the "raw" sentence contains n tokens,
    len(self.tokens) == n+1 and the dependency matrix has (n+1)**2 entries.

    This representation allows the dependency graph to have any structure, not necessarily a tree.
    It can thus represent enhanced UD relations.
    """

    def __init__(self, tokens, dependencies, multiword_tokens=None):
        self.tokens = tokens
        self.dependencies = dependencies
        if multiword_tokens is None:
            self.multiword_tokens = dict()
        else:
            self.multiword_tokens = multiword_tokens

        assert self.tokens[0] == "[root]"
        assert len(self.tokens) == len(self.dependencies)
        assert all(len(self.tokens) == len(row) for row in self.dependencies)

    def __len__(self):
        return len(self.tokens)

    def __str__(self):
        return "DependencyAnnotatedSentence(\"{}\")".format(" ".join(self.tokens[1:]))

    @staticmethod
    def from_conllu(pyconll_sent):
        """Read in a sentence from the representation returned by pyconll. A [root] token will be added to the
        beginning of the sentence.
        """
        # Create token list and ID->Index dictionary
        tokens = ['[root]']
        id_to_ix = dict({'0': 0})
        ix = 1
        multiword_tokens = dict()
        for token in pyconll_sent:
            if "-" in token.id:
                multiword_tokens[ix] = (token.id, token.form)
                continue
            tokens.append(token.form)
            id_to_ix[token.id] = ix
            ix += 1

        # Create dependency matrix
        dependencies = [["[null]" for i in range(len(tokens))] for j in range(len(tokens))]  # Default is no relation
        for token in pyconll_sent:
            for (head_id, (rel_type, rel_subtype1, rel_subtype2, rel_subtype3)) in token.deps.items():
                dependent_ix = id_to_ix[token.id]
                head_ix = id_to_ix[head_id]
                rel_subtype = ":".join(rel_sub for rel_sub in (rel_subtype1, rel_subtype2, rel_subtype3) if rel_sub is not None)
                relation = rel_type if rel_subtype == "" else rel_type + ":" + rel_subtype
                dependencies[head_ix][dependent_ix] = relation

        return DependencyAnnotatedSentence(tokens, dependencies, multiword_tokens)

    @staticmethod
    def from_tensor(tokens, dependencies_tensor, label_vocab):
        """Create a DependencyAnnotatedSentence from a list of tokens and a "flat" tensor containing
        dependency label IDs.
        """
        dep_label_matrix = LabelIndexMatrix.from_tensor(dependencies_tensor, len(tokens))
        dependencies = list()
        for orig_row in dep_label_matrix:
            new_row = list()
            for orig_cell in orig_row:
                new_cell = label_vocab.ix2token[orig_cell]
                new_row.append(new_cell)
            dependencies.append(new_row)

        return DependencyAnnotatedSentence(tokens, dependencies)

    def add_multiword_token(self, index, id, token):
        """Add a "multiword token" to this sentence at the specified index. A MWT does not become part of the relation
        matrix, but is printed out when representing the sentence in CoNLL-U format.
        Example (German): "zum" <--> "zu dem"
        """
        self.multiword_tokens[index] = (id, token)

    def tokens_no_root(self):
        """Return the "raw" tokens of this sentences, i.e. everything except the [root] token at the start."""
        assert self.tokens[0] == "[root]"
        return self.tokens[1:]

    def to_conllu(self):
        """Output a string that contains this annotated sentence in CoNLL-U format.

        For now, only contains word form and dependencies (basic + enhanced) and omits all other information such as POS
        or lemma. The columns for basic dependencies is filled with placeholder material
        """
        conllu_string = ""

        for dependent_ix, token in list(enumerate(self.tokens))[1:]:
            deps = list()
            column = [row[dependent_ix] for row in self.dependencies]
            for head_ix, relation in enumerate(column):
                if relation != "[null]":
                    deps.append((head_ix, relation))

            assert deps  # Every token must have at least one head

            # Construct enhanced column
            enhanced_deps = "|".join("{}:{}".format(head_ix, relation) for (head_ix, relation) in deps)

            # Basic column is placeholder only
            if dependent_ix == 1:
                basic_head, basic_deprel = 0, "root"
            else:
                basic_head, basic_deprel = 1, "dep"

            if dependent_ix in self.multiword_tokens:  # If a MWT comes before the token in question, print it out here
                mwt_id, mwt_form = self.multiword_tokens[dependent_ix]
                conllu_string += "\t".join((str(mwt_id), mwt_form, "_", "_", "_", "_", "_", "_", "_", "_")) + "\n"

            conllu_string += "\t".join((str(dependent_ix), token, "_", "_", "_", "_", str(basic_head), basic_deprel, enhanced_deps, "_")) + "\n"

        return conllu_string

    def dependencies_as_index_matrix(self, label_vocab, padding_index=-1):
        """Convert the dependencies of this sentence to a LabelIndexMatrix."""
        return LabelIndexMatrix.from_label_matrix(self.dependencies, label_vocab, padding_index)

    def pretty_print(self):
        """Display sentence as a nicely formatted dependency matrix."""
        # Determine required column width for printing
        col_width = 0
        for token in self.tokens:
            col_width = max(col_width, len(token))
        for i in range(len(self.dependencies)):
            for j in range(len(self.dependencies[i])):
                if len(self.dependencies[i][j]) > col_width:
                    col_width = max(col_width, len(self.dependencies[i][j]))
        col_width += 3

        # Print dependency matrix
        print()
        print("".join(token.rjust(col_width) for token in [""] + self.tokens))
        print()
        for head_ix in range(len(self.tokens)):
            print(self.tokens[head_ix].rjust(col_width), end="")
            for dependent_ix in range(len(self.tokens)):
                print(self.dependencies[head_ix][dependent_ix].rjust(col_width), end="")
            print()
            print()

    @staticmethod
    def get_parsing_counts(gold, predicted):
        """Compare a system-parsed sentence with the corresponding gold-standard sentence; for each dependency
        label, return the counts for
          (a) how often this label occurred in the gold standard ("gold")
          (b) how often the system predicted this label ("predicted")
          (c) how often the gold label and predicted label were identical ("correct").
        """
        assert len(predicted) == len(gold)
        assert all(tok1 == tok2 for tok1, tok2 in zip(predicted.tokens, gold.tokens))

        counts = defaultdict(lambda: {"predicted": 0, "gold": 0, "correct": 0})

        for i in range(len(predicted)):
            for j in range(len(predicted)):
                predicted_label = predicted.dependencies[i][j]
                gold_label = gold.dependencies[i][j]

                if gold_label != "[null]":
                    counts[gold_label]["gold"] += 1
                if predicted_label != "[null]":
                    counts[predicted_label]["predicted"] += 1
                    if predicted_label == gold_label:
                        counts[predicted_label]["correct"] += 1

        return counts


def heads(sentence, token_ix):
    """For a given token in a sentence (specified via its index), generate all of its syntactic heads, together with
    the dependency relations by which they are attached.

    The order is going outwards from the specified token, first right to left, then left to right.
    """
    for i in chain(range(token_ix - 1, 0, -1), range(token_ix + 1, len(sentence))):  # Iterate to the left from word, then to the right
        deprel = sentence.dependencies[i][token_ix]
        if deprel != "[null]":
            yield i, deprel


def dependents(sentence, token_ix):
    """For a given token in a sentence (specified via its index), generate all of its syntactic dependents, together
    with the dependency relations by which they are attached.

    The order is going outwards from the specified token, first right to left, then left to right.
    """
    for j in chain(range(token_ix - 1, 0, -1), range(token_ix + 1, len(sentence))):
        deprel = sentence.dependencies[token_ix][j]
        if deprel != "[null]":
            yield j, deprel
