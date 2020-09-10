# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Stefan Grünewald

import torch

from torch.utils.data import DataLoader

from .conllu_dataset import CoNLLUDataset


class CONLLULoader(DataLoader):
    """Class for loading batches of sentences from a corpus file in CONLL-U format."""

    def __init__(self, corpus_path, label_vocab, batch_size=10, shuffle=True, num_workers=1):
        self.conllu_dataset = CoNLLUDataset.from_corpus_file(corpus_path)
        self.label_vocab = label_vocab

        super().__init__(self.conllu_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         collate_fn=lambda x: batchify(x, self.label_vocab))


def batchify(sentences, label_vocab):
    """Helper function to create model input / gold output from a bunch of DependencyAnnotatedSentences.

    Output: A tuple whose first element is the list of sentences and whose second element is a tensor containing
    the gold dependency indices. Shape of the latter is (num_sentences, max_sent_length**2).
    """
    # Create gold output (label index matrix tensors)
    dep_matrices = [sent.dependencies_as_index_matrix(label_vocab) for sent in sentences]
    max_len_dep = max(len(dep_matrix) for dep_matrix in dep_matrices)
    for dep_matrix in dep_matrices:
        dep_matrix.tensorize(padded_length=max_len_dep)

    dep_matrix_batch = torch.stack([dep_matrix.data for dep_matrix in dep_matrices])

    return sentences, dep_matrix_batch
