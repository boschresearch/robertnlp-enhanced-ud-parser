# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Stefan Grünewald

"""Script for replacing lexicalized material in dependency relations with placeholders
in a corpus.
"""

import pyconll

from sys import argv

lex_rels = {"obl", "nmod", "advcl", "acl", "conj"}


def delexicalize(corpus_filename):
    raw_sents = pyconll.load_from_file(corpus_filename)

    for sent in raw_sents:
        for token in sent:
            for (head_id, (rel_type, rel_subtype1, rel_subtype2, rel_subtype3)) in token.deps.items():
                if rel_type in lex_rels and rel_subtype1 is not None and rel_subtype1 not in {"poss", "tmod", "npmod", "relcl"}:
                    if rel_type in {"obl", "nmod"}:
                        token.deps[head_id] = (rel_type, "[case]", rel_subtype2, rel_subtype3)
                    if rel_type in {"advcl", "acl"}:
                        token.deps[head_id] = (rel_type, "[mark]", rel_subtype2, rel_subtype3)
                    if rel_type == "conj":
                        token.deps[head_id] = (rel_type, "[cc]", rel_subtype2, rel_subtype3)

        print(sent.conll())
        print()


if __name__ == "__main__":
    corpus_filename = argv[1]
    delexicalize(corpus_filename)
