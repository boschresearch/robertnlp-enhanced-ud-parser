# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Stefan Grünewald

"""Script for repairing structurally invalid dependency graphs.
Steps:
1. Add missing root
2. Connect nodes which are not reachable from root
3. Remove dependency self-loops
"""

import sys
import os
import code
import pyconll

udify_path = sys.argv[1]
udify_archive_path = sys.argv[2]
corpus_filename = os.path.abspath(sys.argv[3])

sys.path.insert(1, udify_path)
os.chdir(udify_path)

from udify import util
from allennlp.predictors.predictor import Predictor


def add_missing_root(sentence, udify_annotations):
    assert len(sentence) == len(udify_annotations["predicted_dependencies"])
    for token, udify_dep in zip(sentence, udify_annotations["predicted_dependencies"]):
        if udify_dep == "root":
            token.deps = {"0": ("root", None, None, None)}


def remove_self_loops(sentence):
    for token in sentence:
        if token.id in token.deps:
            rel = token.deps[token.id][0]
            del(token.deps[token.id])
            if not token.deps:
                token.deps[str(int(token.id) - 1)] = (rel, None, None, None)


def get_reachable_from_root(sentence):
    reachable_from_root = set()
    for root_id in get_root_ids(sentence):
        reachable_from_root |= get_reachable_from(sentence, root_id, set())

    not_reachable_from_root = get_all_ids(sentence) - reachable_from_root
    
    return reachable_from_root, not_reachable_from_root


def get_reachable_from(sentence, node_id, encountered_nodes):
    encountered_nodes.add(node_id)

    for dependent_id in dependents(sentence, node_id):
        if dependent_id not in encountered_nodes:
            encountered_nodes |= get_reachable_from(sentence, dependent_id, encountered_nodes)

    return encountered_nodes


def connect_graph(sentence, reachable_from_root, not_reachable_from_root, udify_annotations):
    # Get UDify annotations
    udify_heads = udify_annotations["predicted_heads"]
    udify_deprels = udify_annotations["predicted_dependencies"]

    # Compute token ID mapping
    id_to_ix = {token.id: ix for ix, token in enumerate(sentence)}

    # Compute reachabilities within the disconnected component(s)
    disconnected_component_roots = set()
    num_descendants = dict()
    for node_id in not_reachable_from_root:
        descendants = get_reachable_from(sentence, node_id, set())
        num_descendants[node_id] = len(descendants)
        if descendants >= not_reachable_from_root:  # Node is "root" of disco	deleted:    data/corpora/all_english/all-english-dev.delex.conllu
nnected component
            disconnected_component_roots.add(node_id)

            token_head = str(udify_heads[id_to_ix[node_id]])
            token_deprel = udify_deprels[id_to_ix[node_id]]
            if token_head in reachable_from_root:  # There is a relation in the UDify basic layer which links the components -> add that one to the enhanced layer
                sentence[node_id].deps[token_head] = (token_deprel, None, None, None)
                return

    sent_root = next(iter(get_root_ids(sentence)))
    if disconnected_component_roots:
        # Pick one of the "roots" of the disconnected component and just connect it
        # to (one of) the root(s) of the full sentence via the placeholder "dep" relation
        arbitrary_node_id = next(iter(disconnected_component_roots))
        sentence[arbitrary_node_id].deps[sent_root] = ("dep", None, None, None)
    else:
        # If there is no "root" of the disconnected component, greedily connect the node with the
        # most descendants and iterate until everything is connected
        best_connected_id = sorted(num_descendants.items(), key=lambda x: x[1], reverse=True)[0][0]
        sentence[best_connected_id].deps[sent_root] = ("dep", None, None, None)
        reachable_from_root, not_reachable_from_root = get_reachable_from_root(sentence)
        connect_graph(sentence, reachable_from_root, not_reachable_from_root, udify_annotations)


def get_all_ids(sentence):
    return {token.id for token in sentence}


def get_root_ids(sentence):
    for token in sentence:
        if "0" in token.deps:
            yield token.id 


def dependents(sentence, node_id):
    for possible_dependent in sentence:
        if node_id in possible_dependent.deps:
            yield possible_dependent.id


if __name__ == "__main__":
    archive = util.load_archive(sys.argv[2])
    udify_predictor = Predictor.from_archive(archive, "udify_predictor")

    sentences = pyconll.load_from_file(corpus_filename)
    for sentence in sentences:
        udify_annotations = None

        if not set(get_root_ids(sentence)):
            udify_annotations = udify_predictor.predict(" ".join([token.form for token in sentence]))
            add_missing_root(sentence, udify_annotations)

        reachable_from_root, not_reachable_from_root = get_reachable_from_root(sentence)       
        if not_reachable_from_root:
            udify_annotations = udify_predictor.predict(" ".join([token.form for token in sentence])) if udify_annotations is None else udify_annotations
            connect_graph(sentence, reachable_from_root, not_reachable_from_root, udify_annotations)

        remove_self_loops(sentence)

        print(sentence.conll())
        print()

